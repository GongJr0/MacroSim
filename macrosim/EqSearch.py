from typing import Optional, Any

from pysr import PySRRegressor
import sympy as sp

from macrosim.Utils import SrConfig, sr_generator

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, KFold

from sklearn.neighbors import LocalOutlierFactor

import pandas as pd
import numpy as np
from dataclasses import replace

VALID_BINARY_OPS = tuple['+', '-', '*', '/', '^']
FULL_BINARY_OPS: VALID_BINARY_OPS = ('+', '-', '*', '/', '^')

DEFAULT_UNARY = {
            'inv': {
                'julia': 'inv(x)=1/x',
                'sympy': lambda x: 1/x
            },
            'safe_log': {
                'julia': 'safe_log(x) = sign(x) * log(abs(x))',
                'sympy': lambda x: sp.sign(x) * sp.log(abs(x))
            },
            'safe_sqrt': {
                'julia': 'safe_sqrt(x) = sign(x) * sqrt(abs(x))',
                'sympy': lambda x: sp.sign(x) * sp.sqrt(abs(x))
            },
            'soft_guard_root': {
                'julia': 'soft_guard_root(x::T) where {T<:Real} = sqrt(sqrt(x^2 + T(1e-6)))',  # 1e-6 = safe epsilon at 32bit precision
                'sympy': lambda x: sp.sqrt(sp.sqrt(x ** 2 + 1e-6))
            },
            'exp': {
                'julia': 'exp',
                'sympy': lambda x: sp.exp(x)
            }
        }
DEFAULT_CONSTRAINTS = {
        '^': (-1, 2),
        'exp': 4,
        'safe_log': 3,
        'safe_sqrt': 2,
        'soft_guard_root': 2,
        'inv': -1
}

DEFAULT_SR_CONFIG_EQ_SEARCH = SrConfig(
                    model_selection='accuracy',
                    maxsize=32,
                    niterations=300,

                    elementwise_loss='L2DistLoss()',

                    binary_operators=list(FULL_BINARY_OPS),
                    unary_operators=[item['julia'] for item in DEFAULT_UNARY.values()],
                    extra_sympy_mappings={item[0]: item[1]['sympy'] for item in DEFAULT_UNARY.items()},

                    constraints=DEFAULT_CONSTRAINTS,

                    random_state=0,
                    deterministic=True,
                    parallelism='serial',

                    verbosity=0,
                    progress=False,
                    temp_equation_file=True
)


class EqSearch:
    """
    Automated symbolic search engine using PySR's cutting-edge SymbolicRegressor model.
    """

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.DataFrame | pd.Series,  # y.shape = (-1, 1)
                 random_state: int = 0,
                 **kwargs) -> None:

        self.extra_unary: dict[str, dict[str, Any]] = {
            'inv': {
                'julia': 'inv(x)=1/x',
                'sympy': lambda x: 1/x
            },
            'safe_log': {
                'julia': 'safe_log(x) = sign(x) * log(abs(x))',
                'sympy': lambda x: sp.sign(x) * sp.log(abs(x))
            },
            'safe_sqrt': {
                'julia': 'safe_sqrt(x) = sign(x) * sqrt(abs(x))',
                'sympy': lambda x: sp.sign(x) * sp.sqrt(abs(x))
            },
            'soft_guard_root': {
                'julia': 'soft_guard_root(x::T) where {T<:Real} = sqrt(sqrt(x^2 + T(1e-6)))',  # 1e-6 = safe epsilon at 32bit precision
                'sympy': lambda x: sp.sqrt(sp.sqrt(x ** 2 + 1e-6))
            },
            'exp': {
                'julia': 'exp',
                'sympy': lambda x: sp.exp(x)
            }
        }

        self.X = X
        self.y = pd.DataFrame(y)

        self.random_state = random_state

        self.distilled = None

        self.sr = PySRRegressor()

    def distil_split(self, test_size: float = 0.2,
                     grid_search: bool = False, gs_params: Optional[dict[str, Any]] = ...) -> None:
        X = self.X.copy()
        y = self.y.copy()

        lof = LocalOutlierFactor(n_neighbors=int(np.floor(len(X)**0.5)), contamination=0.025)
        lof.fit(X)
        outliers = np.where(lof.negative_outlier_factor_ == -1, True, False)

        X['outlier'] = outliers
        y['outlier'] = outliers

        X = X[~X['outlier']].drop('outlier', axis=1)
        y = y[~y['outlier']].drop('outlier', axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size, random_state=self.random_state)

        rf = RandomForestRegressor(n_estimators=100,
                                   random_state=self.random_state)

        if grid_search:
            gs = GridSearchCV(estimator=RandomForestRegressor(), param_grid=gs_params, cv=5)
            gs.fit(X_train, y_train)
            rf = gs.best_estimator_
        else:
            rf.fit(X_train, y_train.values.ravel())

        print(f"RandomForest Score at Distillation: {rf.score(X_test, y_test):.3f}")

        distilled_y = rf.predict(self.X)

        self.distilled = pd.DataFrame(distilled_y, index=self.X.index)

    def search(self,
               extra_unary_ops: Optional[dict[str, dict[str, Any]]] = None,
               custom_loss: Optional[str] = None,
               constraints: Optional[dict[str, tuple[int, int]]] = None,
               cv: int = 1,
               **kwargs) -> None:

        assert self.distilled.shape == self.y.shape, "Run self.distil_split() before symbolizing."
        
        if extra_unary_ops is None:
            extra_unary_ops = {}

        if constraints is None:
            constraints = {}

        unary = DEFAULT_UNARY | extra_unary_ops
        constraints = DEFAULT_CONSTRAINTS | constraints

        cfg = replace(DEFAULT_SR_CONFIG_EQ_SEARCH,
                      unary_operators=[item['julia'] for item in unary.values()],
                      extra_sympy_mappings={item[0]: item[1]['sympy'] for item in unary.items()},
                      constraints=constraints,
                      elementwise_loss=custom_loss or DEFAULT_SR_CONFIG_EQ_SEARCH.elementwise_loss
                      )

        if cv > 1:
            kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

            folds = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

                curr_sr = sr_generator(cfg)

                curr_sr.fit(X_train, y_train)
                fold_mse = mean_squared_error(curr_sr.predict(X_val), y_val)
                folds.append((curr_sr, fold_mse))

            sr = sorted(folds, key=lambda x: x[1])[0][0]
            self.sr = sr

        else:
            curr_sr = sr_generator(cfg)

            curr_sr.fit(self.X, self.y)
            self.sr = curr_sr

    @property
    def get_eq(self) -> sp.Expr:
        return self.sr.get_best()['sympy_format']

    @property
    def get_model(self) -> PySRRegressor:
        return self.sr
