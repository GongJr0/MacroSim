import datetime as dt
import pandas as pd
from fredapi import Fred
from macrosim.stats.Causality import Causality as Causality
from macrosim.stats.Stationarity import Stationarity as Stationarity
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

class SeriesAccessor:
    fred: Fred
    def __init__(self, key_path: Path | str, key_name: str) -> None: ...
    @staticmethod
    def get_key(key_path: Path | str, key_name: str) -> str: ...
    def get_series(self, series_ids: Sequence[str], date_range: tuple[dt.date, dt.date], reindex_freq: str | None = None, *, series_alias: Iterable[str] | Iterable[None] | None = None) -> pd.DataFrame: ...
    def fill(self, data: pd.DataFrame, methods: Sequence[Literal['divide', 'bfill', 'ffill', 'mean', 'median', 'IQR_mean'] | Callable[[pd.Series], pd.Series] | None]) -> pd.DataFrame: ...
    @property
    def FILL_MAP(self): ...
