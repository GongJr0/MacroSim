import numpy as np
import scipy.optimize as opt
import pandas as pd

from typing import Callable


class GrowthPatternDetector:
    def __init__(self):
        ...

    @staticmethod
    def _linear(a, x, c) -> float:
        return a * x + c

    @staticmethod
    def _exponential(x, r) -> float:
        return x * np.exp(r)

    @staticmethod
    def _logarithmic(x, a, c) -> float:
        return a * np.sign(x) * np.log(np.abs(x) + 1) + c  # Avoid non-positive log

    @staticmethod
    def _power_law(a, t, c) -> float:
        return a * (t**c)

    @staticmethod
    def _logistic(x, r, k) -> float:
        return x+ (r*x)*(1 - (x/k))

    @staticmethod
    def fit_pattern(df: pd.DataFrame) -> dict[str, tuple[Callable[[float], float], float]]:
        out = {k: None for k in df.columns}
        x_data = np.arange(1, len(df) + 1)  # Time series index

        for col in df.columns:
            y_data = df[col].values
            best_fit = None
            best_mse = float("inf")

            growth_functions = {
                "linear": (GrowthPatternDetector._linear, [1, 0]),
                "exponential": (GrowthPatternDetector._exponential, [0.01]),
                "logarithmic": (GrowthPatternDetector._logarithmic, [10, -100]),
                "power_law": (GrowthPatternDetector._power_law, [1, 1]),
                "logistic": (GrowthPatternDetector._logistic, [0.1, max(df[col].max(), 1)]),
            }

            for name, (func, p0) in growth_functions.items():
                try:
                    params, _ = opt.curve_fit(func, x_data, y_data, p0=p0, maxfev=5000)
                    y_pred = func(x_data, *params)
                    mse = np.mean((y_data - y_pred) ** 2)

                    if mse < best_mse:
                        best_mse = mse
                        best_fit = (name, params)

                except (RuntimeError, ValueError):
                    continue  # Skip if fitting fails

            if best_fit:
                name, params = best_fit
                chosen_func = lambda x, func=growth_functions[name][0], params=params: func(x, *params)
                out[col] = (chosen_func, best_mse)

        return out

