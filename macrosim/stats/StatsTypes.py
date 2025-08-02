
from enum import Enum
from typing import Union, Literal, NewType, cast, Optional
import pandas as pd

DATA = Union[pd.Series, pd.DataFrame]

FREQ_TO_PERIODS_PER_YEAR: dict[str, int] = {
    "D": 365,
    "B": 252,  # Assumed 252, can change year to year
    "W": 52,
    "M": 12,
    "MS": 12,
    "Q": 4,
    "QS": 4,
    "A": 1,
    "AS": 1,
    "Y": 1,
    "H": 24 * 365,
    "T": 60 * 24 * 365,
    "min": 60 * 24 * 365,
    "S": 60 * 60 * 24 * 365,
}


class PVAL(float):
    alpha: float

    def __new__(cls, val, alpha=0.05):
        obj = super().__new__(cls, val)
        obj.alpha = alpha
        return obj

    @property
    def reject(self) -> bool:
        return self < self.alpha


LAG = NewType('LAG', int)


class TestStats(Enum):
    F = 'ssr_ftest'
    chi2 = 'ssr_chi2test'
    lr = 'lrtest'


class SeriesInfo(Enum):
    CAUSAL = "CAUSAL"
    NON_CAUSAL = "NON_CAUSAL"

    STATIONARY = "STATIONARY"
    NON_STATIONARY = "NON_STATIONARY"
