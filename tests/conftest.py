import pytest
import pandas as pd
import datetime as dt
import os
from pandas.core.indexes.datetimes import DatetimeIndex  # type: ignore

from macrosim import SeriesAccessor, AutoReg


@pytest.fixture(scope="module")
def df():
    path = './mock_data.csv'
    if not os.path.exists(path):
        fred = SeriesAccessor(
            key_path='./fred_key.env',
            key_name='FRED_KEY'
        )
        start_date = dt.datetime(2018, 1, 1)
        end_date = dt.datetime(2025, 1, 1)
        data = fred.get_series(series_ids=['GDP', 'M2V', 'QBPBSTLKTEQKTBKEQKCOMSTK', 'LRAC64TTUSQ156S'],
                               date_range=(start_date, end_date))

        data.index.name = 'Date'
        data.to_csv(path)

    df_out = pd.read_csv(path)
    df_out.set_index('Date', inplace=True)
    df_out.index = pd.to_datetime(df_out.index)
    assert isinstance(df_out.index, DatetimeIndex)
    return df_out


@pytest.fixture(scope="module")
def t_range() -> pd.DatetimeIndex:
    return pd.date_range(start='2018-01-01', end='2020-01-01')


@pytest.fixture(scope="module")
def autoreg(df: pd.DataFrame) -> AutoReg:
    return AutoReg(
        df=df,
        target='GDP'
    )
