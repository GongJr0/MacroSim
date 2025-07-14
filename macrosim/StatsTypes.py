
from enum import Enum
from typing import Union, Literal, NewType, cast, Optional
import pandas as pd

DATA = Union[pd.Series, pd.DataFrame]


class PVAL(float):
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
