from pysr import PySRRegressor

import numpy as np
import sympy as sp
import pandas as pd

from macrosim.Utils import DataUtils as du
from macrosim.Utils import SrConfig, sr_generator
from dataclasses import replace

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import LocalOutlierFactor

from statsmodels.tsa.stattools import acf
from scipy.signal import periodogram

from typing import Optional, Any, Callable

EXTRA_UNARY_DEFAULT = {
        'inv': {
            'julia': 'inv(x)=1/x',
            'sympy': lambda x: 1/x
        },

        'atan': {
            'julia': 'atan',
            'sympy': lambda x: sp.atan(x)
        }
    }

EXTRA_UNARY_SEASONAL = {
        'sin': {
            'julia': 'sin',
            'sympy': lambda x: sp.sin(x)
        },

        'sin2': {
            'julia': 'sin2(x)=sin(x)^2',
            'sympy': lambda x: sp.sin(x)**2
        },

        'tan': {
            'julia': 'tan',
            'sympy': lambda x: sp.tan(x)
        },

        'tan2': {
            'julia': 'tan2(x)=tan(x)^2',
            'sympy': lambda x: sp.tan(x)**2
        }
    }

UNARY_DEFAULT = {
        'safe_log': {
            'julia': 'safe_log(x) = sign(x) * log(abs(x))',
            'sympy': lambda x: sp.sign(x) * sp.log(abs(x))
        },
        'safe_sqrt': {
            'julia': 'safe_sqrt(x) = sign(x) * sqrt(abs(x))',
            'sympy': lambda x: sp.sign(x) * sp.sqrt(abs(x))
        },

        'soft_guard_root': {
            'julia': 'soft_guard_root(x::T) where {T<:Real} = sqrt(sqrt(x^2 + T(1e-6)))',
            'sympy': lambda x: sp.sqrt(sp.sqrt(x**2 + 1e-6))
        },

        'exp': {
            'julia': 'exp',
            'sympy': lambda x: sp.exp(x)
        }
    }


BINARY_DEFAULT = ['+', '-', '/', '*', '^']

CONSTRAINTS_DEFAULT = {
        '^': (-1, 3),
        'atan': 3,
        'safe_log': 3,
        'safe_sqrt': 2,
        'soft_guard_root': 2,
        'inv': -1
    }

CONSTRAINTS_SEASONAL = {
        'sin': 2,
        'sin2': 2,
        'tan': 2,
        'tan2': 2
    }

DEFAULT_SR_CONFIG_BASE = SrConfig(
                model_selection='accuracy',  # type:ignore

                niterations=300,
                maxsize=32,

                binary_operators=BINARY_DEFAULT,
                unary_operators=[item['julia'] for item in (UNARY_DEFAULT | EXTRA_UNARY_DEFAULT).values()],
                extra_sympy_mappings={item[0]: item[1]['sympy'] for item in (UNARY_DEFAULT | EXTRA_UNARY_DEFAULT).items()},

                constraints=CONSTRAINTS_DEFAULT,
                elementwise_loss='L2DistLoss()',
                progress=False,
                temp_equation_file=True,

                deterministic=True,
                random_state=0,
                parallelism='serial',  # type:ignore
                batching=False,
                verbosity=0,
            )


class BaseVarModel:

    def __init__(self,
                 var: pd.Series,
                 lag_count_descriptor: Optional[Callable[[pd.Series], int]] = None):
        self.var = var

        self.sr: PySRRegressor | None = None
        self.rf: RandomForestRegressor | None = None

        self.sr_loss: int | None = None
        self.rf_loss: int | None = None

        self.n_lags_func = lag_count_descriptor if lag_count_descriptor is not None else self.DEFAULT_LAG_COUNT
        self.filtered = du.lof_filter(self.var)

    def get_seasonal_freq(self) -> int | None:
        # Compute ACF with a reasonable maximum lag (half of the series length)
        max_lag = int(max(len(self.var) / 2, 2))
        acf_values = acf(self.var.values, nlags=max_lag)

        acf_series = pd.Series(acf_values).pct_change().abs()
        significant_deltas = acf_series[acf_series >= 0.1]

        def average_freq(lst: pd.Series) -> int | None:
            if len(lst) < 2:
                return None  # Not seasonal if 1 spike in ACF deltas
            periods = [lst.iloc[i + 1] - lst.iloc[i] for i in range(len(lst) - 1)]
            return round(sum(periods) / len(periods))

        if not significant_deltas.empty:
            return average_freq(significant_deltas)
        else:
            return None

    @staticmethod
    def is_seasonal(series: pd.Series) -> bool:

        if len(series) < 24:  # High likelihood of overfitting for short series
            return False
        window = int(np.ceil(len(series)**(1/3)))

        detrended = series - pd.Series(series).rolling(window=window, center=True).mean()
        detrended = detrended.dropna()

        autocorr = acf(detrended, fft=True, nlags=min(100, len(detrended) // 2))
        acf_peaks = (autocorr[1:] > 0.5).sum()  # Ignore lag 0, count strong spikes

        freqs, power = periodogram(detrended, scaling='spectrum')
        power_threshold = np.mean(power) + 3 * np.std(power)  # type:ignore
        dominant_peaks = (power > power_threshold).sum()

        return acf_peaks >= 1 and dominant_peaks >= 1

    def rf_distil(self,
                  grid_search: bool = False,
                  gs_params: Optional[dict[str, list[Any]]] = None):

        lagged_df = du.get_lags(self.filtered)

        rf = RandomForestRegressor(n_estimators=300, random_state=0)
        X = lagged_df.drop('X_t', axis=1)
        y = lagged_df['X_t']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        if grid_search:
            gscv = GridSearchCV(estimator=rf, param_grid=gs_params, cv=5, n_jobs=-1)
            gscv.fit(X_train, y_train)
            rf = gscv.best_estimator_
        else:
            rf.fit(X_train, y_train)

        score = rf.score(X_test, y_test)
        print(f"RandomForest score at distillation: {score:.3f}")

        if score >= 0.9:
            print('Distillation has acceptable accuracy.')
        else:
            print('Distillation is rejected due to low accuracy.')

        self.rf = rf
        self.rf_loss = mean_squared_error(y_test, rf.predict(X_test))

        return pd.Series(rf.predict(X), name='X_t', index=y.index) if score >= 0.9 else y

    def symbolic_model(self, cv=5, **kwargs) -> sp.Expr:
        lof_filtered = self.filtered
        is_seasonal = self.is_seasonal(self.var)

        if is_seasonal:
            unary = UNARY_DEFAULT | EXTRA_UNARY_DEFAULT | EXTRA_UNARY_SEASONAL
            constraints = CONSTRAINTS_DEFAULT | CONSTRAINTS_SEASONAL

            cfg = replace(DEFAULT_SR_CONFIG_BASE,
                          constraints=constraints,
                          unary_operators=[item['julia'] for item in unary.values()],
                          extra_sympy_mappings={item[0]: item[1]['sympy'] for item in unary.items()}
                          )
            print(f"Cyclical trig functions are enabled for {self.var.name} due to detected seasonality.")

        else:
            cfg = DEFAULT_SR_CONFIG_BASE
            print(f"No seasonality detected in {self.var.name}. Cyclical trig functions are disabled.")

        lagged_df = du.get_lags(lof_filtered)
        X_t = self.rf_distil(grid_search=kwargs.get('grid_search', False),
                             gs_params=kwargs.get('gs_params', {})).to_frame()

        X_lag = lagged_df.drop('X_t', axis=1)

        folds = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=0)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_t.index)):
            X_train, X_val = X_lag.iloc[train_idx], X_lag.iloc[test_idx]
            y_train, y_val = X_t.iloc[train_idx], X_t.iloc[test_idx]

            curr_sr = sr_generator(cfg)

            curr_sr.fit(X_train, y_train)
            fold_mse = mean_squared_error(curr_sr.predict(X_val), y_val)
            folds.append((curr_sr, fold_mse))

        sorted_folds = sorted(folds, key=lambda x: x[1])
        sr = sorted_folds[0][0]
        self.sr = sr
        self.sr_loss = sorted_folds[0][1]

        return sr.get_best()['sympy_format']

    def model_select(self, loss_diff_threshold=0.05) -> PySRRegressor | RandomForestRegressor:
        assert (self.sr_loss is not None) and (self.rf_loss is not None), ("Run symbolic_model to store the best SR "
                                                                           "and RF instances before model selection.")

        loss_diff = (self.sr_loss / self.rf_loss) - 1
        if loss_diff >= loss_diff_threshold:
            selected = self.rf
            print(f"SR Model introduces {loss_diff:.2%} higher MSE compared to RF predictions. "
                  f"Falling back to RF predictions.")
        else:
            selected = self.sr
            print(f"SR did not introduce a significant MSE increase ({loss_diff:.2%}) compared to RF predictions. "
                  "Using Symbolic expressions as the base predictor.")

        return selected

    @property
    def DEFAULT_LAG_COUNT(self) -> Callable[[pd.Series], int]:
        # Cube root of data length
        return lambda x: int(np.ceil(len(x)**(1/3)))
