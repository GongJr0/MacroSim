from macrosim.SeriesAccessor import SeriesAccessor
from macrosim.BaseVarSelector import BaseVarSelector
from macrosim.BaseVarModel import BaseVarModel

import pandas as pd
import numpy as np
import datetime as dt

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


bvm = BaseVarModel(df=df)
candidates = bvm.get_base_candidates()

bvm.symbolic_model(candidate=candidates['POPGROWTH'], progress=True)