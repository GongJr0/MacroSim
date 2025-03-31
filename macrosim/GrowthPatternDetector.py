import pandas as pd
import numpy as np

import scipy.optimize as opt
from typing import Callable


class GrowthPatternDetector:
    def __init__(self) -> None:
        ...  # Static class

    @staticmethod
    def linear_func_opt(series, a, b) -> np.array:
        n = len(series)
        i0 = series.iloc[0]

        out = [i0]
        for _ in range(1, n):
            out.append(a*out[-1] + b)
        return np.array(out)

    @staticmethod
    def linear_func(x, a, b) -> float:
        return a*x + b

    @staticmethod
    def logarithmic_func_opt(series, a, b) -> np.array:
        n = len(series)
        i0 = series.iloc[0]
        out = [i0]

        for _ in range(1, n):
            out.append(a * np.log(out[-1]) + b)
        return np.array(out)

    @staticmethod
    def logarithmic_func(x, a, b) -> float:
        return a * np.log(x) + b

    @staticmethod
    def exponential_func_opt(series, b, c) -> np.array:
        n = len(series)
        i0 = series.iloc[0]
        out = [i0]

        for _ in range(1, n):
            out.append(np.exp(b*out[-1])+c)
        return np.array(out)

    @staticmethod
    def exponential_func(x, b, c) -> float:
        return np.exp(b * x) + c

    @staticmethod
    def logistic_func_opt(series, a, b, c, d) -> np.array:
        n = len(series)
        i0 = series.iloc[0]
        out = [i0]

        for _ in range(1, n):
            out.append(
                a / (1 + np.exp(-b * (out[-1] - c))) + d
            )
        return np.array(out)

    @staticmethod
    def logistic_func(x, a, b, c, d) -> float:
        return a / (1 + np.exp(-b * (x - c))) + d

    @staticmethod
    def gompertz_func_opt(series, a, b, c, d) -> np.array:
        n = len(series)
        i0 = series.iloc[0]
        out = [i0]

        for _ in range(1, n):
            out.append(
                a * np.exp(-b * np.exp(-c * out[-1])) + d
            )

        return np.array(out)

    @staticmethod
    def gompertz_func(x, a, b, c, d) -> float:
        return a * np.exp(-b * np.exp(-c * x)) + d

    def _linear_optimize(self, series: pd.Series) -> tuple[Callable[[float], float], float]:
        # Drop first index of series (output of growth functions are X_{i+1})
        y_true = series.values
        p0_args = [1, 0]  # i.e. 0-growth

        def objective(w):
            pred = self.linear_func_opt(series, *w)
            return ((pred-y_true)**2).mean()

        out = opt.minimize(objective, np.array(p0_args))
        mse = ((self.linear_func_opt(series, *out.x) - y_true) ** 2).mean()

        def fitted_func() -> Callable[[float], float]:
            return lambda x, func=self.linear_func, params=out["x"]: func(x, *params)

        return fitted_func(), mse

    def _logarithmic_optimize(self, series: pd.Series) -> tuple[Callable[[float], float], float]:
        y_true = series.values
        p0_args = [1, 0]  # Initial parameters for a and b

        def objective(w):
            pred = self.logarithmic_func_opt(series, *w)
            return ((pred-y_true)**2).mean()

        out = opt.minimize(objective, np.array(p0_args))
        mse = ((self.logarithmic_func_opt(series, *out.x) - y_true) ** 2).mean()

        def fitted_func() -> Callable[[float], float]:
            return lambda x, func=self.logarithmic_func, params=out["x"]: func(x, *params)

        return fitted_func(), mse

    def _exponential_optimize(self, series: pd.Series) -> tuple[Callable[[float], float], float]:
        y_true = series.values
        p0_args = [1, 0]

        def objective(w):
            pred = self.exponential_func_opt(series, *w)
            return ((pred-y_true)**2).mean()

        out = opt.minimize(objective, np.array(p0_args))
        mse = ((self.exponential_func_opt(series, *out.x) - y_true) ** 2).mean()

        def fitted_func() -> Callable[[float], float]:
            return lambda x, func=self.exponential_func, params=out["x"]: func(x, *params)

        return fitted_func(), mse

    def _logistic_optimize(self, series: pd.Series) -> tuple[Callable[[float], float], float]:
        y_true = series.values
        p0_args = [1, 1, 0.5, 0]

        def objective(w):
            pred = self.logistic_func_opt(series, *w)
            return ((pred-y_true)**2).mean()

        out = opt.minimize(objective, np.array(p0_args))
        mse = ((self.logistic_func_opt(series, *out.x) - y_true) ** 2).mean()

        def fitted_func() -> Callable[[float], float]:
            return lambda x, func=self.logistic_func, params=out["x"]: func(x, *params)

        return fitted_func(), mse

    def _gompertz_optimize(self, series: pd.Series) -> tuple[Callable[[float], float], float]:
        y_true = series.values
        p0_args = [1, 1, 1, 1]

        def objective(w):
            pred = self.gompertz_func_opt(series, *w)
            return ((pred-y_true)**2).mean()

        out = opt.minimize(objective, np.array(p0_args))
        mse = ((self.gompertz_func_opt(series, *out.x) - y_true) ** 2).mean()

        def fitted_func() -> Callable[[float], float]:
            return lambda x, func=self.gompertz_func, params=out["x"]: func(x, *params)

        return fitted_func(), mse
