from statsmodels.tsa.stattools import grangercausalitytests as granger  # type: ignore # noqa

from itertools import permutations
from collections import Counter
from typing import Literal, cast, Optional
from .StatsTypes import FREQ_TO_PERIODS_PER_YEAR, DATA, LAG, PVAL, TestStats, SeriesInfo
from dataclasses import dataclass

import pandas as pd
from pandas.api.extensions import register_dataframe_accessor

import numpy as np
import warnings

"""
Causality is assessed through Granger Causality tests, using the F statistic by default.
max_lags are weighted by a factor of lend(df)/(10*observation_frequency) to prioritize the attainment of 
decent degrees of freedom.

Batch testing is used to assess aggregate causality over portions of the sample, spanning the entire dataset.
batch_size is set to 4*observation_frequency by default, where each batch covers 4 years of data. If the dataset is
not large enough, batching will not be used, and the entire dataset will be tested at once.
"""


class Causality:
    def __init__(self) -> None:
        # Static Class
        ...

    @staticmethod
    def get_freq(df: DATA) -> int:
        assert isinstance(df.index, pd.DatetimeIndex), "DatetimeIndex is mandatory for frequency inference."
        assert isinstance((freq_full := pd.infer_freq(df.index)), str), (
            "Frequency inference failed. Ensure the index is"
            " a DatetimeIndex with a consistent frequency.")

        freq_base = freq_full.split('-')[0]  # Get the base frequency (e.g., 'D', 'M', 'Q', etc.)
        freq = FREQ_TO_PERIODS_PER_YEAR[freq_base]
        return freq

    @staticmethod
    def get_batch_size(data: DATA) -> int:
        freq = Causality.get_freq(data)
        if (bsize := 4*freq) > len(data):
            warnings.warn(f"Consider increasing observation count if possible. "
                          f"{len(data)} observations found. "
                          f"There's not enough observations to use the default batch size of 4*df_frequency. "
                          f"Using single batch instead.",
                          category=UserWarning)
            bsize = len(data)

        return bsize

    @staticmethod
    def max_lag(data: DATA) -> int:
        freq = Causality.get_freq(data)
        return ((len(data)/(20*freq))*np.log(len(data))).round()

    @staticmethod
    def _gct(x: DATA, y: DATA, stat: Literal['F', 'chi2', 'lr'], alpha=0.05) -> list[tuple[LAG, PVAL]]:
        assert x.shape[0] == y.shape[0], 'X and Y must have the same number of rows.'
        assert stat in ['F', 'chi2', 'lr'], f"Invalid stat: {stat}. Must be 'F', 'chi2', or 'lr'."

        data = pd.concat([x, y], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            test = granger(data, maxlag=Causality.max_lag(data), verbose=False)

        causal = []
        for lag, result in test.items():
            val = PVAL(result[0][TestStats[stat].value][1], alpha=alpha)
            if val.reject:
                causal.append((cast(LAG, lag), val))

        return causal

    @staticmethod
    def perm_gct(df: pd.DataFrame, stat: Literal['F', 'chi2', 'lr'] = 'F', alpha=0.05) -> dict[str, dict[str, list[tuple[LAG, PVAL]]]]:
        pairs = permutations(df.columns, 2)
        causality_matrix: dict[str, dict[str, list[tuple[LAG, PVAL]]]] = {
            col: {} for col in df.columns
        }

        for pair in pairs:
            causality_matrix[pair[1]][pair[0]] = Causality._gct(df[pair[0]], df[pair[1]], stat, alpha=alpha)

        return causality_matrix

    @staticmethod
    def rolling_perm_gct(
            df: DATA,
            stat: Literal['F', 'chi2', 'lr'] = 'F',
            alpha: float = 0.05,
            batch_size: Optional[int] = None
    ) -> dict[str, dict[str, list[tuple[LAG, PVAL]]]]:

        out: dict[str, dict[str, list[tuple[LAG, PVAL]]]] = {col: {} for col in df.columns}  # type: ignore
        if batch_size is None:
            batch_size = Causality.get_batch_size(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            batched = np.array_split(df, len(df) // batch_size)

        # Run causality tests on batches
        for batch in batched:

            batch_result = Causality.perm_gct(cast(pd.DataFrame, batch), stat, alpha)

            for col, inner_result in batch_result.items():
                for inner_col, results in inner_result.items():
                    if inner_col not in out[col]:
                        out[col][inner_col] = []
                    out[col][inner_col].extend(results)

        # Aggregate all lag results per (col, inner_col) pair
        for col, inner in out.items():
            for inner_col, res in inner.items():
                df_lags = pd.DataFrame(res, columns=["lag", "pval"])
                grouped: pd.DataFrame = df_lags.groupby("lag", as_index=False).agg({'pval': 'mean'})  # type: ignore
                aggregate_lag_pvals: list[tuple[LAG, PVAL]] = []
                for _, row in grouped.iterrows():
                    lag = cast(LAG, row['lag'])
                    pval = PVAL(row['pval'], alpha=alpha)
                    aggregate_lag_pvals.append((cast(LAG, int(lag)), pval))

                out[col][inner_col] = aggregate_lag_pvals

        return out

    @staticmethod
    def common_lag(gct_res: dict[str, dict[str, list[tuple[LAG, PVAL]]]]) -> LAG:

        out = [
            t[0]
            for inner_dict in gct_res.values()
            for lst in inner_dict.values()
            for t in lst
        ]
        counter = Counter(out)
        common_lag = min(
            counter.items(),
            key=lambda x: (-x[1], x[0])  # sort by freq descending, then int ascending
        )[0]
        return common_lag


@dataclass
class CausalityResult:
    df: DATA

    def __post_init__(self):
        self.rolling = Causality.rolling_perm_gct(self.df)
        self.tests = Causality.perm_gct(self.df)
        self.is_causal = {
            col: SeriesInfo.CAUSAL if any(inner for inner in self.tests[col].values()) else SeriesInfo.NON_CAUSAL
            for col in self.tests.keys()
        }
        self.common_lag = Causality.common_lag(self.tests)


@register_dataframe_accessor("causality")
class CausalityAccessor:
    def __init__(self, pandas_obj: DATA):
        self._obj = pandas_obj
        self._results: CausalityResult = None  # type: ignore

    def _compute_results(self):
        self._results = CausalityResult(self._obj)

    @property
    def results(self) -> CausalityResult:
        if not self._results:
            self._compute_results()

        return self._results

    @property
    def tests(self) -> dict[str, dict[str, list[tuple[LAG, PVAL]]]]:
        if not self._results:
            self._compute_results()

        return self._results.tests

    @property
    def rolling(self) -> dict[str, dict[str, list[tuple[LAG, PVAL]]]]:
        if not self._results:
            self._compute_results()

        return self._results.rolling

    @property
    def is_causal(self) -> dict[str, Literal[SeriesInfo.CAUSAL, SeriesInfo.NON_CAUSAL]]:
        if not self._results:
            self._compute_results()

        return self._results.is_causal

    @property
    def common_lag(self) -> LAG:
        if not self._results:
            self._compute_results()
        return self._results.common_lag
