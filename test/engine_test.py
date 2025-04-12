import datetime as dt
import pickle

from macrosim.SeriesAccessor import SeriesAccessor
from macrosim.SimEngine import SimEngine
from macrosim.EqSearch import EqSearch
from macrosim.GrowthDetector import GrowthDetector


fred = SeriesAccessor(
    key_path='../fred_key.env',
    key_name='FRED_KEY'
)

start = dt.datetime.fromisoformat('2002-01-01')
end = dt.datetime.fromisoformat('2024-01-01')

df = fred.get_series(
    series_ids=['NETEXP', 'CIVPART', 'CORESTICKM159SFRBATL', 'LES1252881600Q', 'SPPOPGROWUSA', 'A264RX1A020NBEA', 'GDPC1'],
    series_alias=[None, None, 'CPI', 'RWAGE', 'POPGROWTH', 'I_C', 'RGDP'],
    reindex_freq='QS',
    date_range=(start, end),

)
df = fred.fill(
    data=df,
    methods=[None, None, None, None, 'ffill', 'divide', None]
)

eqsr = EqSearch(
    X=df.drop('RGDP', axis=1),
    y=df['RGDP']
)
eqsr.distil_split()
eqsr.search()

main_estimator = eqsr.get_model

gd = GrowthDetector(features=df.drop('RGDP', axis=1))
estimators = gd.compose_estimators()

init_params = {}
for col in eqsr.X:
    data = eqsr.X[col].tail(gd.get_lag_count)
    estimator = estimators[col]
    init_params[col] = (data, estimator)


engine = SimEngine(
    sr=main_estimator,
    n_lags=gd.get_lag_count,
    init_params=init_params,
    entropy_coef=None
)

out = engine.simulate(50)
with open('out.pkl', 'wb') as f:
    pickle.dump(out, f)