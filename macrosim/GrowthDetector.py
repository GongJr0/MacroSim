from sklearn.ensemble import RandomForestRegressor

from macrosim.BaseVarSelector import BaseVarSelector
from macrosim.BaseVarModel import BaseVarModel
from macrosim.Utils import SrConfig, sr_generator
from macrosim.Utils import DataUtils as du

from pysr import PySRRegressor
from sklearn.neighbors import LocalOutlierFactor

import pandas as pd
import numpy as np

import sympy as sp

import pickle
from joblib import Parallel, delayed

from dataclasses import replace

DEFAULT_SR_CONFIG_NON_BASE = SrConfig(
    # Search method config
    model_selection='accuracy',
    maxsize=16,
    niterations=100,

    # Operations config
    binary_operators=['+', '-', '*', '/', '^'],
    unary_operators=['exp',
                     'safe_log(x) = sign(x) * log(abs(x))',
                     'safe_sqrt(x) = sign(x) * sqrt(abs(x))',
                     'soft_guard_root(x::T) where {T<:Real} = sqrt(sqrt(x^2 + T(1e-6)))',
                     'inv(x)=1/x'],
    extra_sympy_mappings={
        'inv': lambda x: 1/x,
        'safe_log': lambda x: sp.sign(x) * sp.log(abs(x)),
        'safe_sqrt': lambda x: sp.sign(x) * sp.sqrt(abs(x)),
        'soft_guard_root': lambda x: sp.sqrt(sp.sqrt(x**2 + 1e-6)),
    },

    # Constraints config
    constraints={
        '^': (-1, 2),
        'exp': 4,
        'safe_log': 3,
        'safe_sqrt': 2,
        'soft_guard_root': 2,
        'inv': -1
    },

    # Loss config
    elementwise_loss='L2DistLoss()',

    # Search Deterministic Behavior Config
    deterministic=True,
    parallelism='serial',
    random_state=0,

    # Misc Params
    temp_equation_file=True,
    progress=False,
    batching=False,
    verbosity=0
)


class GrowthEstimator:
    def __init__(self, model, is_base: bool = False):
        self.model = model
        self.is_base = is_base

    def __getattr__(self, attr):
        # Delegate attribute access to the wrapped model
        return getattr(self.model, attr)

    def __setattr__(self, key, value):
        # Allow setting `model` and `is_base` normally
        if key in {"model", "is_base"}:
            object.__setattr__(self, key, value)
        else:
            setattr(self.model, key, value)

    def __repr__(self):
        return f"WrappedRegressor(is_base={self.is_base}, model={repr(self.model)})"


class GrowthDetector:
    def __init__(self, features: pd.DataFrame) -> None:

        self.df = features
        self.vars = features.columns.values

        self.bvs = BaseVarSelector(features)
        self.base = self.bvs.get_base_candidates()
        self.base_vars = self.base.columns.values
        self.non_base_vars = [var for var in self.vars if var not in self.base_vars]

        self._base_kwargs = {}
        self._non_base_kwargs = {}

        self.base_estimators: dict[str, GrowthEstimator | None] = {
            k: None for k in self.base.columns
        }

        self.estimators: dict[str, GrowthEstimator | None] = {
            k: None for k in self.vars
        }

    def _get_base_var_growth(self, cv=5, **kwargs) -> None:
        base_df = self.base
        var_ls = self.base_vars

        def fit_variable(var, df):
            series = df[var]
            bvm = BaseVarModel(series)

            bvm.symbolic_model(cv=cv, **kwargs)
            fitted_estimator = bvm.model_select()

            sr_params = None
            if isinstance(fitted_estimator, PySRRegressor):
                sr_params = bvm.sr.get_params()
                out = fitted_estimator.get_best().to_frame().T
            elif isinstance(fitted_estimator, RandomForestRegressor):
                out = fitted_estimator

            return var, out, sr_params
        results = Parallel(n_jobs=-1)(
            delayed(fit_variable)(var, base_df) for var in var_ls
        )

        for var, out, sr_params in results:
            if not isinstance(out, RandomForestRegressor):
                feature_names = [f"X_t{n}" for n in range(1, self._n_lags(base_df[var])+1)]
                label_name = 'X_t'
                dummy_frame = pd.DataFrame(
                    np.zeros((1, len(feature_names) + 1)),  # 1 row, N+1 columns
                    columns=[label_name, *feature_names]
                )

                sr = PySRRegressor()
                sr.set_params(**sr_params)
                sr.set_params(maxsize=7, niterations=1, verbosity=0)  # type:ignore

                sr.fit(dummy_frame.drop(label_name, axis=1), dummy_frame[label_name])
                sr.equations_ = out

                estimator = GrowthEstimator(sr, is_base=True)

            elif isinstance(out, RandomForestRegressor):
                estimator = GrowthEstimator(out, is_base=True)

            self.base_estimators[var] = estimator
            self.estimators[var] = estimator

    def _get_non_base_var_growth(self, **kwargs) -> None:
        base = self.base

        def fit_non_base_var(var, df):
            cfg = replace(DEFAULT_SR_CONFIG_NON_BASE, **(kwargs or {}))
            sr = sr_generator(config=cfg)

            filtered = du.lof_filter(df[var])

            lags = du.get_lags(filtered)
            base_index_matched = base.loc[lags.index, :]

            X = pd.concat([lags.drop('X_t', axis=1), base_index_matched], axis=1)
            y = lags['X_t']

            sr.fit(X, y)
            out = sr.get_best().to_frame().T
            return var, out

        results = Parallel(n_jobs=-1)(
            delayed(fit_non_base_var)(var, self.df)
            for var in self.vars if var not in self.base_vars
        )

        for var, out in results:
            feature_names = [*[f"X_t{n}" for n in range(1, self._n_lags(self.df[var]) + 1)], *self.base_vars]
            label_name = 'X_t'
            dummy_frame = pd.DataFrame(
                np.zeros((1, len(feature_names) + 1)),  # 1 row, N+1 columns
                columns=[label_name, *feature_names]
            )

            sr = sr_generator(
                SrConfig(maxsize=7, niterations=1, verbosity=0)
            )

            sr.fit(dummy_frame.drop(label_name, axis=1), dummy_frame[label_name])
            sr.equations_ = out

            estimator = GrowthEstimator(sr, is_base=False)
            self.estimators[var] = estimator

    def base_estimator_kwargs(self, **kwargs) -> None:
        self._base_kwargs = kwargs

    def non_base_estimator_kwargs(self, **kwargs) -> None:
        self._non_base_kwargs = kwargs

    def compose_estimators(self, cv=5) -> dict[str, GrowthEstimator]:

        self._get_base_var_growth(cv, **self._base_kwargs)
        self._get_non_base_var_growth(**self._non_base_kwargs)

        return self.estimators

    # WIP
    def serialize_estimators(self, file: str = "growth_estimators.pkl") -> None:
        with open(file, 'wb') as f:
            dump = (self, self.estimators)
            pickle.dump(dump, f)

    @staticmethod
    def _n_lags(series):
        return int(np.ceil(len(series)**(1/3)))

    @property
    def IS_BASE(self):
        return {
            k: True if k in self.base_vars else False for k in self.vars
        }

    @property
    def get_lag_count(self) -> int:
        return self._n_lags(self.df)
    