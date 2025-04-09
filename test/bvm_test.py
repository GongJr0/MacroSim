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

bvs = BaseVarSelector(df)
base = bvs.get_base_candidates()
print(f"{base.columns[1]} is being modelled...")

bvm = BaseVarModel(base.iloc[:, 1])

bvm.symbolic_model(progress=True, maxsize=16, niterations=100, constraints={'atan': 2})
if bvm.model_select() == bvm.sr:
    print(f"SR Loss: {bvm.sr_loss}")
    print(f"RF Loss: {bvm.rf_loss}")
    print(bvm.sr.get_best())
