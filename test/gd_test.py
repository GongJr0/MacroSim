import pandas as pd
import numpy as np
import datetime as dt

from macrosim.SeriesAccessor import SeriesAccessor
from macrosim.GrowthDetector import GrowthDetector

fred = SeriesAccessor(
    key_path='../fred_key.env',
    key_name='FRED_KEY'
)

start = dt.datetime.fromisoformat('2002-01-01')
end = dt.datetime.fromisoformat('2024-01-01')

df = fred.get_series(
    series_ids = ['NETEXP', 'CIVPART', 'CORESTICKM159SFRBATL', 'LES1252881600Q', 'SPPOPGROWUSA', 'A264RX1A020NBEA', 'GDPC1'],
    series_alias=[None, None, 'CPI', 'RWAGE', 'POPGROWTH', 'I_C', 'RGDP'],
    reindex_freq='QS',
    date_range=(start, end),

)
df = fred.fill(
    data=df,
    methods=[None, None, None, None, 'ffill', 'divide', None]
)

gd = GrowthDetector(df.drop('RGDP', axis=1))
out = gd.compose_estimators(cv=2)
print(out)