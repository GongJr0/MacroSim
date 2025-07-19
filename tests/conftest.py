import pytest
import pandas as pd
import datetime as dt
import os

from macrosim import SeriesAccessor


@pytest.fixture(scope="module")
def df():
    # Replaces your __main__ block, runs once per test module
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
    return df_out
