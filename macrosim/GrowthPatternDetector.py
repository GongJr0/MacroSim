import pandas as pd
import numpy as np

import scipy.optimize as opt
from typing import Callable


def exp_func(x, a, b) -> float:
    return a * x + b


def linear_func(x, c) -> float:
    return x+c


def log_func(x, a, r, c) -> float:
    return a*x**r + c


class GrowthPatternDetector:
    def __init__(self) -> None:
        ...  # Static class

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
            pred = self.serialize(series, exp_func, *w)
            return ((pred-y_true)**2).mean()

        out = opt.minimize(objective, np.array(p0_args))
        mse = ((self.serialize(series, exp_func, *out.x) - y_true) ** 2).mean()

        def fitted_func() -> Callable[[float], float]:
            return lambda x, func=exp_func, params=out["x"]: func(x, *params)

        return fitted_func(), mse

    def _linear_optimize(self, series: pd.Series) -> tuple[Callable[[float], float], float]:
        y_true = series.values
        p0_args = [series.pct_change().mean() * series.mean()]

        def objective(w):
            pred = self.serialize(series, linear_func, *w)
            return ((pred-y_true)**2).mean()

        out = opt.minimize(objective, np.array(p0_args))
        mse = ((self.serialize(series, linear_func, *out.x) - y_true) ** 2).mean()

        def fitted_func() -> Callable[[float], float]:
            return lambda x, func=linear_func, params=out["x"]: func(x, *params)


        return fitted_func(), mse

    def _log_optimize(self, series: pd.Series) -> tuple[Callable[[float], float], float]:
        y_true = series.values
        p0_args = np.array([1, 0.5, 0])

        def objective(w):
            pred = self.serialize(series, log_func, *w)
            return ((pred-y_true)**2).mean()

        out = opt.minimize(objective, p0_args,
                           bounds=[(None, None), (1e-10, 1), (None, None)])

        mse = ((self.serialize(series, log_func, *out.x) - y_true) ** 2).mean()

        def fitted_func() -> Callable[[float], float]:
            return lambda x, func=log_func, params=out["x"]: func(x, *params)


        return fitted_func(), mse

    def find_opt_growth(self, df: pd.DataFrame) -> dict[str, tuple[Callable[[float], float], float]]:
        out = {}
        for col in df.columns:
            search = [func(df[col]) for func in self.opt_funcs]
            best = sorted(search, key=lambda x: x[1])[0]
            out[col] = best

        return out

    @property
    def opt_funcs(self) -> list:
        return [
            self._exp_optimize,
            self._linear_optimize,
            self._log_optimize,
        ]