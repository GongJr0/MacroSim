from statsmodels.tsa.stattools import grangercausalitytests as granger  # type: ignore # noqa

import warnings
from itertools import permutations
from collections import defaultdict, Counter
from typing import Literal, cast, Any
from macrosim.StatsTypes import DATA, LAG, PVAL, TestStats, SeriesInfo
from dataclasses import dataclass
from copy import deepcopy

import pandas as pd
from pandas.api.extensions import register_dataframe_accessor

import numpy as np
import warnings

class Causality:
    def __init__(self) -> None:
        # Static Class
        ...

    @staticmethod
    def max_lag(data: DATA) -> int:
        return int(np.ceil(
            np.sqrt(data.shape[0])
        ))

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
            alpha=0.05,
            batch_size=30
    ) -> dict[str, dict[str, list[tuple[LAG, PVAL]]]]:

        out: dict[str, dict[str, list[tuple[LAG, PVAL]]]] = {col: {} for col in df.columns}  # type: ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            batched = np.array_split(df, len(df) // batch_size)

        # Run causality tests on batches
        for batch in batched:
            batch_result = Causality.perm_gct(batch, stat, alpha)

            for col, inner_result in batch_result.items():
                for inner_col, results in inner_result.items():
                    if inner_col not in out[col]:
                        out[col][inner_col] = []
                    out[col][inner_col].extend(results)

        # Aggregate all lag results per (col, inner_col) pair
        for col, inner in out.items():
            for inner_col, res in inner.items():
                df_lags = pd.DataFrame(res, columns=["lag", "pval"])
                grouped = df_lags.groupby("lag", as_index=False).agg({'pval': 'mean'})

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
        self.tests = Causality.rolling_perm_gct(self.df)
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
