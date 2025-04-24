from pysr import PySRRegressor

from macrosim.GrowthDetector import GrowthDetector
from macrosim.SeriesAccessor import SeriesAccessor
from macrosim.EqSearch import EqSearch
from macrosim.SimEngine import SimEngine

import datetime as dt
import pickle

fred = SeriesAccessor(
    key_path='../fred_key.env',
    key_name='FRED_KEY'
)

start = dt.datetime.fromisoformat('2002-01-01')
end = dt.datetime.fromisoformat('2024-12-31')

df = fred.get_series(
    series_ids=['LRAC64TTUSQ156S', 'CPIAUCSL', 'M2SL', 'GDPC1'],
    series_alias=['LABPART', 'CPI', 'M2', 'RGDP'],
    date_range=(start, end),
    reindex_freq='QS'
)

gd = GrowthDetector(
    features=df.drop('RGDP', axis=1)
)
growth = gd.compose_estimators(cv=2)

print(growth['LABPART'].get_best()['sympy_format'])
print(growth['CPI'].get_best()['sympy_format'])

# eqsr = EqSearch(
#     X=df.drop('RGDP', axis=1),
#     y=df['RGDP']
# )
# eqsr.distil_split()
# eqsr.search()
#
# init_params = {
#     var: (df[var].head(gd.get_lag_count), growth[var]) for var in df.drop('RGDP', axis=1).columns
# }
# engine = SimEngine(
#     sr=eqsr.get_model,
#     init_params=init_params,
#     n_lags=gd.get_lag_count
# )
#
# out = engine.simulate(50)
#
# with open('out.pkl', 'wb') as f:
#     pickle.dump(out, f)  # type:ignore
