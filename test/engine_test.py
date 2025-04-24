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
    series_ids=['PCECC96', 'GPDIC1', 'EXPGSC1', 'IMPGSC1', 'GDPC1'],
    series_alias=['CONSUMPTION', 'DOMINV', 'REXP', 'RIMP', 'RGDP'],
    date_range=(start, end),
    reindex_freq='QS'
)

gd = GrowthDetector(
    features=df.drop('RGDP', axis=1)
)
growth = gd.compose_estimators(cv=2)

eqsr = EqSearch(
    X=df.drop('RGDP', axis=1),
    y=df['RGDP']
)
eqsr.distil_split()
eqsr.search(maxsize=16, niterations=250)

init_params = {
    var: (df[var].head(gd.get_lag_count), growth[var]) for var in df.drop('RGDP', axis=1).columns
}
engine = SimEngine(
    sr=eqsr.get_model,
    init_params=init_params,
    n_lags=gd.get_lag_count
)

out = engine.simulate(50)

with open('out.pkl', 'wb') as f:
    pickle.dump(out, f)  # type:ignore
