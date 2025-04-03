from fredapi import Fred
import pandas as pd
import numpy as np
import datetime as dt

from dotenv import load_dotenv
from os import getenv, PathLike
from pathlib import Path

from typing import Union, Sequence, Optional, Literal, Callable, Any


class SeriesAccessor:

    def __init__(self, key_path: Union[Path, str], key_name: str):
        key = self.get_key(key_path, key_name)
        self.fred = Fred(api_key=key)

        del key

    @staticmethod
    def get_key(key_path: Union[Path, str], key_name: str) -> str:
        key_path = Path(key_path)  # Ensure it's a Path object
        assert key_path.is_file(), f"File not found: {key_path}"
        assert key_path.suffix == '.env', "Use a .env file"

        load_dotenv(str(key_path))

        key = getenv(key_name)
        assert key, f"Key '{key_name}' not found in {key_path}"

        return key

    def get_series(self,
                   series_ids: Sequence[str],
                   date_range: tuple[dt.date, dt.date],
                   reindex_freq: Optional[str] = None,
                   *,
                   series_alias: Optional[Sequence[str]] = None) -> pd.DataFrame:

        _out = []
        _freq = []

        freq_rank = {
            "L": 0, "S": 1, "T": 2, "H": 3, "BH": 4, "D": 5, "B": 6, "W": 7, "SM": 8,
            "SMS": 9, "BM": 10, "BMS": 11, "MS": 12, "Q": 13, "BQ": 14, "QS": 15, "BQS": 16,
            "A": 17, "BA": 18, "AS": 19, "BAS": 20
        }

        for series_id, alias in zip(series_ids, series_alias):
            series = self.fred.get_series(series_id, observation_start=date_range[0], observation_end=date_range[1])
            series.index = pd.to_datetime(series.index)
            series.name = alias if alias else series_id

            inferred_freq = pd.infer_freq(series.index)
            _freq.append(inferred_freq if inferred_freq else "MS")
            _out.append(series)

        max_freq = min(_freq, key=lambda x: freq_rank.get(x, float("inf")))

        desired_freq = reindex_freq if reindex_freq else max_freq
        new_index = pd.date_range(start=date_range[0], end=date_range[1], freq=desired_freq)

        reindexed_series = [s.reindex(new_index) for s in _out]
        df = pd.concat(reindexed_series, axis=1)

        return df

    def fill(self,
             data: pd.DataFrame,
             methods: Sequence[Union[Literal['divide', 'bfill', 'ffill', 'mean', 'median', 'IQR_mean'], Callable[[pd.Series], pd.Series]]]) -> pd.DataFrame:
        df = data.copy()
        if len(methods) == len(df.columns):
            methods = [*methods, *[None] * (len(df.columns) - len(methods))]

        for col, method in zip(df.columns, methods):
            if method is None:
                continue
            elif isinstance(method, str):
                df[col] = self.FILL_MAP[method](df[col])
            else:
                df[col] = method(df[col])
        return df

    @staticmethod
    def _fill_equal(series: pd.Series) -> pd.Series:
        mask = series.isna()
        cumsum = (~mask).cumsum()
        count_per_group = mask.groupby(cumsum).transform('count')  # Count NaNs before each value

        # Divide known value equally among NaNs + itself
        filled_values = series.ffill() / (count_per_group + 1)

        # Preserve index and replace only NaN values
        return series.where(~mask, filled_values)

    @staticmethod
    def _bfill(series: pd.Series) -> pd.Series:
        return series.bfill(inplace=False)

    @staticmethod
    def _ffill(series: pd.Series) -> pd.Series:
        return series.ffill(inplace=False)

    @staticmethod
    def _mean(series: pd.Series) -> pd.Series:
        return series.mean()

    @staticmethod
    def _median(series: pd.Series) -> pd.Series:
        return series.median()

    @staticmethod
    def _IQR_mean(series: pd.Series) -> pd.Series:
        return series.loc[(series <= series.quantile(0.75)) & (series >= series.quantile(0.25))].mean()

    @property
    def FILL_MAP(self):
        return {
            'divide': self._fill_equal,
            'bfill': self._bfill,
            'ffill': self._ffill,
            'mean': self._mean,
            'median': self._median,
            'IQR_mean': self._IQR_mean
        }