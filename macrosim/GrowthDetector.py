import pandas as pd
import numpy as np

import scipy.optimize as opt
from typing import Callable

import sympy as sp


class NamedFunction:
    def __init__(self, func: Callable[[float], float], params:list[float], name: str):
        self.func = func
        self.name = name
        self.params = params

    def __call__(self, x: float) -> float:
        return self.func(x)

    def __repr__(self):
        return self.name  # This will be shown when printing the dict


class MSE(float):
    def __new__(cls, value):
        return super().__new__(cls, value)  # Ensure MSE behaves like a float

    def __repr__(self):
        return f"MSE = {self:.2f}"  # Custom string representation


def exp_func(x, a, b) -> float:
    return a * x + b


def exp_viz(x, a, b) -> float:
    return sp.sign(x) * a * sp.Abs(x) + b


def linear_func(x, c) -> float:
    return x+c


def log_func(x, a, b, r, c) -> float:
    return a * np.sign(x) * np.log(1+b*np.abs(x)**r)


def log_viz(x, a, b, r, c) -> float:
    return a * sp.sign(x) * sp.log(1+b*sp.Abs(x)**r)


class GrowthDetector:
    def __init__(self) -> None:
        self._out = None

    @staticmethod
    def serialize(series, func, *args) -> np.array:
        n = len(series)
        i0 = series.iloc[0]

        out = [i0]
        for _ in range(1, n):
            out.append(func(out[-1], *args))

        return np.array(out)

    def _exp_optimize(self, series: pd.Series) -> tuple[Callable[[float], float], float]:
        # Drop first index of series (output of growth functions are X_{i+1})
        y_true = series.values
        p0_args = [1, 0]  # i.e. 0-growth

        def objective(w):
            pred = self.serialize(series, exp_func, *w)[:-1]
            return ((pred-y_true[1:])**2).mean()

        out = opt.minimize(objective, np.array(p0_args),
                           bounds=[(1+1e-8, None), (None, None)])
        mse = ((self.serialize(series, exp_func, *out.x)[:-1] - y_true[1:]) ** 2).mean()

        def fitted_func() -> Callable[[float], float]:
            return NamedFunction(lambda x, func=exp_func, params=out["x"]: func(x, *params),
                                 params=out.x,
                                 name=f'Exponential(x, {', '.join([str(i.round(2)) for i in out.x])})')

        return fitted_func(), MSE(mse)

    def _linear_optimize(self, series: pd.Series) -> tuple[Callable[[float], float], float]:
        y_true = series.values
        p0_args = [series.pct_change().mean() * series.mean()]

        def objective(w):
            pred = self.serialize(series, linear_func, *w)[:-1]
            return ((pred-y_true[1:])**2).mean()

        out = opt.minimize(objective, np.array(p0_args))
        mse = ((self.serialize(series, linear_func, *out.x)[:-1] - y_true[1:]) ** 2).mean()

        def fitted_func() -> Callable[[float], float]:
            return NamedFunction(lambda x, func=linear_func, params=out["x"]: func(x, *params),
                                 params=out.x,
                                 name=f'Linear(x, {', '.join([str(i.round(2)) for i in out.x])})')


        return fitted_func(), MSE(mse)

    def _log_optimize(self, series: pd.Series) -> tuple[Callable[[float], float], float]:
        y_true = series.values
        p0_args = np.array([1, 1, 0.5, 0])

        def objective(w):
            pred = self.serialize(series, log_func, *w)[:-1]
            return ((pred-y_true[1:])**2).mean()

        out = opt.minimize(objective, p0_args,
                           bounds=[(None, None), (0, None), (1e-8, 1-1e-8), (None, None)])

        mse = ((self.serialize(series, log_func, *out.x)[:-1] - y_true[1:]) ** 2).mean()

        def fitted_func() -> Callable[[float], float]:
            return NamedFunction(lambda x, func=log_func, params=out["x"]: func(x, *params),
                                 params=out.x,
                                 name=f'Logarithmic(x, {', '.join([str(i.round(2)) for i in out.x])})')


        return fitted_func(), MSE(mse)

    def find_opt_growth(self, df: pd.DataFrame) -> dict[str, tuple[Callable[[float], float], float]]:
        out = {}
        for col in df.columns:
            search = [func(df[col]) for func in self.opt_funcs]
            best = sorted(search, key=lambda x: x[1])[0]
            out[col] = best

        self._out = out
        return out

    @property
    def opt_funcs(self) -> list:
        return [
            self._exp_optimize,
            self._linear_optimize,
            self._log_optimize,
        ]

    @property
    def viz_funcs(self) -> list:
        return [
            exp_viz,
            linear_func,
            log_viz
        ]

    @property
    def sympy_visualize(self) -> dict[str, sp.Expr]:
        x = sp.symbols('x')
        viz = {}
        for k in self._out.keys():
            fun = self._out[k][0]
            if 'Exponential' in fun.__repr__():
                viz[k] = self.viz_funcs[0](x, *fun.params).evalf(n=2)
            elif 'Linear' in fun.__repr__():
                viz[k] = self.viz_funcs[1](x, *fun.params).evalf(n=2)
            elif 'Logarithmic' in fun.__repr__():
                viz[k] = self.viz_funcs[2](x, *fun.params).evalf(n=2)
        return viz