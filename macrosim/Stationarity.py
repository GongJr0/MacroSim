from statsmodels.tsa.stattools import adfuller

import pandas as pd
from pandas.api.extensions import register_dataframe_accessor

from collections import Counter
from dataclasses import dataclass

from .StatsTypes import DATA, LAG, PVAL, SeriesInfo
from typing import cast, Optional, Literal

import numpy as np


class Stationarity:
    def __init__(self):
        # Static Class
        ...

    @staticmethod
    def max_lag(df: DATA) -> LAG:
        return cast(LAG,
                    int(
                        np.ceil(np.sqrt(df.shape[0]))
                    ))

    @staticmethod
    def adf(series: DATA) -> tuple[str, dict[int, PVAL]]:
        assert series.ndim == 1, f"A one dimensional series is necessarry for adfuller. Passed input has {series.ndim} columns."
        assert isinstance(series.name, str), f"Please do not use non string column names. Current name type: {type(series.name)}."

        # adfuller does not run tests for each lag a manual loop with autolag=None will be set up.
        # This functionality is necessarry to determine the lag param which results in the max amount of critical p-values.
        out: dict[int, PVAL] = {}
        for lag in range(1, Stationarity.max_lag(series)+1):
            result: tuple = cast(tuple, adfuller(series, maxlag=lag, autolag=None))
            out[lag] = PVAL(result[1])
        return cast(str, series.name), out

    @staticmethod
    def frame_adf(df: DATA) -> dict[str, dict[int, PVAL]]:
        result = {}
        for col in df.columns:
            test = Stationarity.adf(df[col])
            result[test[0]] = test[1]
        return result

    @staticmethod
    def common_lag(adf_result: dict[str, dict[int, PVAL]]) -> LAG:
        critical_lags =[]
        for res in adf_result.values():
            critical_lags = [*critical_lags, *[k for k, v in res.items() if v.reject]]

        counter = Counter(critical_lags)
        common_lag = min(
            counter.items(),
            key=lambda x: (-x[1], x[0])  # sort by freq descending, then int ascending
        )[0]
        return common_lag


@dataclass
class StationarityResult:
    df: DATA

    def __post_init__(self):
        self.tests = Stationarity.frame_adf(self.df)
        self.is_stationary = {
            col: SeriesInfo.STATIONARY if any(v.reject for v in self.tests[col].values()) else SeriesInfo.NON_STATIONARY
            for col in self.tests.keys()
        }
        self.common_lag = Stationarity.common_lag(self.tests)


@register_dataframe_accessor("stationarity")
class StationarityAccessor:
    def __init__(self, pandas_obj: DATA):
        self._obj = pandas_obj
        self._results: Optional[StationarityResult] = None

    def _compute_results(self):
        if not self._results:
            self._results = StationarityResult(self._obj)

    @property
    def results(self):
        return self._results

    @property
    def is_stationary(self) -> dict[str, Literal[SeriesInfo.STATIONARY, SeriesInfo.NON_STATIONARY]]:
        if not self._results:
            self._compute_results()
        return self._results.is_stationary

    @property
    def common_lag(self) -> LAG:
        if not self._results:
            self._compute_results()
        return self._results.common_lag
