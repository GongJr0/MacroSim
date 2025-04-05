from pip._internal.models import candidate

from macrosim.BaseVarSelector import BaseVarSelector

from pysr import PySRRegressor

import numpy as np
import sympy as sp
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

from statsmodels.tsa.stattools import acf


class BaseVarModel(BaseVarSelector):

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.df = df

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

        sr = PySRRegressor(
            model_selection=kwargs.get('model_selection', 'accuracy'),  # type:ignore

            niterations=kwargs.get('niterations', 300),
            maxsize=kwargs.get('maxsize', 32),

            binary_operators=self.BINARY_DEFAULT,
            unary_operators=[item['julia'] for item in unary.values()],
            extra_sympy_mappings={x[0]: x[1]['sympy'] for x in unary.items()},

            constraints=constraints,
            elementwise_loss=kwargs.get('elementwise_loss', 'L2DistLoss()'),
            progress=kwargs.get('progress', False),
            temp_equation_file=kwargs.get('temp_equation_file', True),

            deterministic=kwargs.get('deterministic', True),
            random_state=kwargs.get('random_state', 0),
            parallelism=kwargs.get('parallelism', 'serial')  # type:ignore
        )

        lagged_df = lof_filtered.to_frame()
        lagged_df.columns = ['X_t']

        lags = int(np.floor(np.sqrt(len(lagged_df))))
        for lag in range(1, lags + 1):
            lagged_df[f"X_t{lag}"] = lagged_df['X_t'].shift(lag)

        lagged_df = lagged_df.dropna(how='any')
        X_t = lagged_df['X_t']
        X_lag = lagged_df.drop('X_t', axis=1)

        sr.fit(X=X_lag,
               y=X_t)

        return sr.get_best()['sympy_format']

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
        return {'^': (-1, 3)}

    @property
    def CONSTRAINTS_SEASONAL(self):
        return {
            'sin': 2,
            'sin2': 2,
            'tan': 2,
            'tan2': 2
        }
