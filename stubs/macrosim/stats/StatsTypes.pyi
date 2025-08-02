import pandas as pd
from _typeshed import Incomplete
from enum import Enum

DATA = pd.Series | pd.DataFrame
FREQ_TO_PERIODS_PER_YEAR: dict[str, int]

class PVAL(float):
    alpha: float
    def __new__(cls, val, alpha: float = 0.05): ...
    @property
    def reject(self) -> bool: ...

LAG: Incomplete

class TestStats(Enum):
    F = 'ssr_ftest'
    chi2 = 'ssr_chi2test'
    lr = 'lrtest'

class SeriesInfo(Enum):
    CAUSAL = 'CAUSAL'
    NON_CAUSAL = 'NON_CAUSAL'
    STATIONARY = 'STATIONARY'
    NON_STATIONARY = 'NON_STATIONARY'
