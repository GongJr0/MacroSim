from pysr import PySRRegressor

from macrosim.GrowthDetector import GrowthDetector
from macrosim.SeriesAccessor import SeriesAccessor
from macrosim.EqSearch import EqSearch
from macrosim.SimEngine import SimEngine

import datetime as dt
import pickle
import time

s_time = time.time()
fred = SeriesAccessor(
    key_path='../fred_key.env',
    key_name='FRED_KEY'
)

start = dt.datetime.fromisoformat('2002-01-01')
end = dt.datetime.fromisoformat('2024-01-01')

df = fred.get_series(
    series_ids=['CIVPART', 'IMPGSC1', 'UNRATE', 'GPDI', 'CORESTICKM159SFRBATL', 'GDPC1'],
    series_alias=[None, 'RIMP', 'UNEMP', 'DOMINV', 'CPI', 'RGDP'],
    reindex_freq='QS',
    date_range=(start, end)
)

print(df.isna().sum())

gd = GrowthDetector(
    features=df.drop('RGDP', axis=1)
)
gd.base_estimator_kwargs(verbosity=0, niterations=250)
gd.non_base_estimator_kwargs(verbosity=0, niterations=250)

growth = gd.compose_estimators(cv=2)

eqsr = EqSearch(
    X=df.drop('RGDP', axis=1),
    y=df['RGDP']
)
eqsr.distil_split()
eqsr.search(maxsize=32, niterations=250, verbosity=1,
            constraints={
                'safe_log': -1,
                'safe_sqrt': -1,
                'soft_guard_root': -1,
                'atan': -1,
                'inv': -1,
                'exp': -1,
                '^': (-1, 3)
            })

init_params = {
    var: (df[var].head(gd.get_lag_count), growth[var]) for var in df.drop('RGDP', axis=1).columns
}
engine = SimEngine(
    sr=eqsr.get_model,
    init_params=init_params,
    n_lags=gd.get_lag_count
)

out = engine.simulate(50)
e_time = time.time()

print(f"Elapsed time: {e_time - s_time:.2f} seconds")

with open('out.pkl', 'wb') as f:
    pickle.dump(out, f)  # type:ignore
