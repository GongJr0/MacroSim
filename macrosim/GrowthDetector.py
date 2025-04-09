from macrosim.BaseVarSelector import BaseVarSelector
from macrosim.BaseVarModel import BaseVarModel

from pysr import PySRRegressor
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np

import scipy.optimize as opt
from typing import Callable, Any

import sympy as sp

import pickle

class GrowthDetector:
    def __init__(self, features: pd.DataFrame) -> None:

        self.df = features
        self.vars = features.columns

        self.bvs = BaseVarSelector(features)
        self.base = self.bvs.get_base_candidates()
        self.base_vars = self.base.columns.values

        self.base_estimators = {
            k: None for k in self.base.columns
        }

        self.estimators: dict[str, PySRRegressor | RandomForestRegressor | None] = {
            k: None for k in self.vars
        }

    def get_lags(self, series) -> pd.DataFrame:
        lagged_df = series.to_frame()
        lagged_df.columns = ['X_t']

        lags = self.n_lags(series)
        for lag in range(1, lags + 1):
            lagged_df[f"X_t{lag}"] = lagged_df['X_t'].shift(lag)

        lagged_df = lagged_df.dropna(how='any')
        return lagged_df


    def get_base_var_growth(self, cv=5) -> None:
        base = self.base
        vars = self.base_vars

        for var in vars:
            series = base[var]
            bvm = BaseVarModel(series)

            bvm.symbolic_model(cv=cv)
            estimator = bvm.model_select()
            self.base_estimators[var] = estimator
            self.estimators[var] = estimator

    def get_non_base_var_growth(self) -> None:
        base = self.base

        for var in self.vars:
            if var not in self.base_vars:
                estimator = self.sr_generator()

                lags = self.get_lags(self.df[var])
                base_index_matched = base.loc[lags.index, :]

                X = pd.concat([lags.drop('X_t', axis=1), base_index_matched], axis=1)
                y = lags['X_t']

                estimator.fit(X, y)
                self.estimators[var] = estimator

    def compose_estimators(self, cv=5) -> dict[str, PySRRegressor | RandomForestRegressor]:
        self.get_base_var_growth(cv)
        self.get_non_base_var_growth()

        return self.estimators

    def serialize_estimators(self, file: str = "growth_estimators.pkl") -> None:
        with open(file, 'wb') as f:
            dump = (self, self.estimators)
            pickle.dump(dump, f)

    @staticmethod
    def n_lags(series):
        return int(np.ceil(len(series)**(1/3)))

    @staticmethod
    def sr_generator():
        sr = PySRRegressor(
            # Search method config
            model_selection='best',
            maxsize=16,
            niterations=100,

            # Operations config
            binary_operators=['+', '-', '*', '/', '^'],
            unary_operators=['exp', 'log', 'sqrt', 'inv(x)=1/x'],
            extra_sympy_mappings={'inv': lambda x: 1/x},

            # Constraints config
            constraints={
                '^': {-1, 2},
                'exp': 4,
                'log': 4,
                'sqrt': 2,
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
            batching=True

        )
        return sr


    @property
    def IS_BASE(self):
        return {
            k: True if k in self.base_vars else False for k in self.vars
        }
