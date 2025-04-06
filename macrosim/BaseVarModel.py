from macrosim.BaseVarSelector import BaseVarSelector

from pysr import PySRRegressor

import numpy as np
import sympy as sp
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import LocalOutlierFactor

from statsmodels.tsa.stattools import acf
from typing import Optional, Any

class BaseVarModel(BaseVarSelector):

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.df = df
        self.sr: PySRRegressor | None = None
        self.rf: RandomForestRegressor | None = None

        self.sr_loss: int | None = None
        self.rf_loss: int | None = None

    def get_base_candidates(self) -> pd.DataFrame:
        self.granger_matrix(score=True)
        self.multivar_granger_matrix()

        scores = self.score_dict
        agg_scores = {k: sum(scores[k].values()) for k in scores.keys()}
        sorted_scores = sorted(agg_scores.items(), key=lambda x: x[1])

        candidates = [score[0] for score in sorted_scores][:2]
        return self.df[candidates]

    @staticmethod
    def lof_filter(candidate: pd.Series):
        n = int(np.floor(np.sqrt(len(candidate))))
        lof = LocalOutlierFactor(n_neighbors=n, contamination='auto')
        lof_mask = np.where(lof.fit_predict(candidate.to_frame()) == -1, True, False)

        return candidate[lof_mask]

    @staticmethod
    def get_seasonal_freq(candidate: pd.Series):
        # Compute ACF with a reasonable maximum lag (half of the series length)
        max_lag = int(max(len(candidate) / 2, 2))  # Convert to integer for valid max_lag
        acf_values = acf(candidate.values, nlags=max_lag)

        acf_series = pd.Series(acf_values).pct_change().abs()
        significant_deltas = acf_series[acf_series >= 0.1]

        if not significant_deltas.empty:
            return significant_deltas.index[0]
        else:
            return None

    def is_seasonal(self, candidate: pd.Series):
        freq_est = self.get_seasonal_freq(candidate)
        cycle_est = np.floor(len(candidate) / freq_est)

        min_strong_lag = int(len(candidate) / 2)

        autocorr = acf(candidate, nlags=min_strong_lag)
        strong_lags = (np.abs(autocorr) >= 0.1).sum()

        return strong_lags >= min_strong_lag

    @staticmethod
    def get_lags(series: pd.Series) -> pd.DataFrame:
        lagged_df = series.to_frame()
        lagged_df.columns = ['X_t']

        lags = int(np.floor(np.sqrt(len(lagged_df))))
        for lag in range(1, lags + 1):
            lagged_df[f"X_t{lag}"] = lagged_df['X_t'].shift(lag)

        lagged_df = lagged_df.dropna(how='any')
        return lagged_df

    def rf_distil(self, lagged_df: pd.DataFrame, grid_search: bool = False, gs_params: Optional[dict[str, list[Any]]] = None):

        rf = RandomForestRegressor(random_state=0)
        X = lagged_df.drop('X_t', axis=1)
        y = lagged_df['X_t']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        if grid_search:
            gscv = GridSearchCV(estimator=rf, param_grid=gs_params, cv=5, n_jobs=-1)
            gscv.fit(X_train, y_train)
            rf = gscv.best_estimator_
        else:
            rf.fit(X_train, y_train)

        print(f"RandomForest score at distillation: {rf.score(X_test, y_test):.3f}")
        self.rf = rf
        self.rf_loss = mean_squared_error(y_test, rf.predict(X_test))

        return pd.Series(rf.predict(X), name='X_t', index=y.index)

    def symbolic_model(self, candidate: pd.Series, **kwargs) -> sp.Expr:
        lof_filtered = self.lof_filter(candidate)
        is_seasonal = self.is_seasonal(lof_filtered)

        unary = self.EXTRA_UNARY_DEFAULT
        constraints = self.CONSTRAINTS_DEFAULT

        if is_seasonal:
            print("Detected seasonal autocorrelation. Cyclical trig functions will be included in symbolic search.")
            unary = unary | self.EXTRA_UNARY_SEASONAL
            constraints = constraints | self.CONSTRAINTS_SEASONAL
        else:
            print("No seasonal behavior detected. Cyclical trig functions will be excluded in symbolic search.")

        lagged_df = self.get_lags(lof_filtered)
        X_t = self.rf_distil(lagged_df,
                             grid_search=kwargs.get('grid_search', False),
                             gs_params=kwargs.get('gs_params', {})).to_frame()

        X_lag = lagged_df.drop('X_t', axis=1)

        folds = []
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_t.index)):
            X_train, X_val = X_lag.iloc[train_idx], X_lag.iloc[test_idx]
            y_train, y_val = X_t.iloc[train_idx], X_t.iloc[test_idx]

            curr_sr = PySRRegressor(
                model_selection=kwargs.get('model_selection', 'accuracy'),  # type:ignore

                niterations=kwargs.get('niterations', 300),
                maxsize=kwargs.get('maxsize', 32),

                binary_operators=self.BINARY_DEFAULT,
                unary_operators=[item['julia'] for item in unary.values()],
                extra_sympy_mappings={x[0]: x[1]['sympy'] for x in unary.items()},

                constraints=constraints | kwargs.get('constraints', {}),
                elementwise_loss=kwargs.get('elementwise_loss', 'L2DistLoss()'),
                progress=kwargs.get('progress', False),
                temp_equation_file=kwargs.get('temp_equation_file', True),

                deterministic=kwargs.get('deterministic', True),
                random_state=kwargs.get('random_state', 0),
                parallelism=kwargs.get('parallelism', 'serial')  # type:ignore
            )

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
            print(f"SR Model Introduces {loss_diff:.2%} more MSE compared to RF predictions. "
                  f"Falling back to RF predictions.")
        else:
            selected = self.sr
            print("SR did not introduce a significant MSE increase compared to RF predictions. "
                  "Using Symbolic expressions as the base predictor.")

        return selected

    @property
    def EXTRA_UNARY_DEFAULT(self):
        return {
            'inv': {
                'julia': 'inv(x)=1/x',
                'sympy': lambda x: 1/x
            },

            'atan': {
                'julia': 'atan',
                'sympy': lambda x: sp.atan(x)
            }
        }

    @property
    def EXTRA_UNARY_SEASONAL(self):
        return {
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

    @property
    def UNARY_DEFAULT(self):
        return ['exp', 'log', 'sqrt']

    @property
    def BINARY_DEFAULT(self):
        return ['+', '-', '/', '*', '^']

    @property
    def CONSTRAINTS_DEFAULT(self):
        return {
            '^': (-1, 3)
        }

    @property
    def CONSTRAINTS_SEASONAL(self):
        return {
            'sin': 2,
            'sin2': 2,
            'tan': 2,
            'tan2': 2
        }
