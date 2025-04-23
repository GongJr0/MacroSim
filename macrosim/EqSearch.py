from typing import Optional, Any

from pysr import PySRRegressor
import sympy as sp

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, KFold

from sklearn.neighbors import LocalOutlierFactor

import pandas as pd
import numpy as np

VALID_BINARY_OPS = tuple['+', '-', '*', '/', '^']
FULL_BINARY_OPS: VALID_BINARY_OPS = ('+', '-', '*', '/', '^')

DEFAULT_UNARY = tuple()

class EqSearch:
    """
    Automated symbolic search engine using PySR's cutting-edge SymbolicRegressor model.
    """

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.DataFrame | pd.Series,  # y.shape = (-1, 1)
                 random_state: int = 0) -> None:

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
                'julia': 'soft_guard_root(x) = sqrt(sqrt(x^2 + 1e-8))',
                'sympy': lambda x: sp.sqrt(sp.sqrt(x ** 2 + 1e-8))
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

    @staticmethod
    def get_new_sr(**kwargs) -> PySRRegressor:
        return PySRRegressor(
                    model_selection=kwargs.get('model_selection', 'accuracy'),  # type:ignore # Do not consider complexity at selection

                    maxsize=kwargs.get('maxsize', 32),  # type:ignore
                    niterations=kwargs.get('niterations', 300),  # type:ignore

                    verbosity=kwargs.get('verbosity', 1),  # type:ignore
                    progress=kwargs.get('progress', False),  # type:ignore
                    temp_equation_file=kwargs.get('temp_equation_file', True),  # type:ignore
                )

    def search(self,
               binary_ops: VALID_BINARY_OPS = FULL_BINARY_OPS,
               unary_ops=DEFAULT_UNARY,
               extra_unary_ops: Optional[dict[str, dict[str, Any]]] = None,
               custom_loss: Optional[str] = None,
               constraints: Optional[dict[str, tuple[int, int]]] = None,
               cv: bool = False,
               **kwargs) -> None:
        
        if extra_unary_ops is None:
            extra_unary_ops = {}

        if constraints is None:
            constraints = {}

        assert self.distilled.shape == self.y.shape, "Run self.distil_split() before symbolizing."

        extra_unary = self.extra_unary | extra_unary_ops

        binary_ops = list(binary_ops)
        unary_ops = list(unary_ops)

        if cv:
            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

            folds = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

                curr_sr = self.get_new_sr(**kwargs)
                curr_sr.set_params(
                    binary_operators=binary_ops,  # type:ignore
                    unary_operators=[*unary_ops, *[x['julia'] for x in extra_unary.values()]],  # type:ignore
                    extra_sympy_mappings={x[0]: x[1]['sympy'] for x in extra_unary.items()},  # type:ignore

                    elementwise_loss=custom_loss if custom_loss else 'L2DistLoss()',  # type:ignore

                    constraints={'^': (-1, 1)} | constraints,
                    random_state=self.random_state,  # type:ignore
                    deterministic=True,  # type:ignore
                    parallelism='serial'  # type:ignore
                )

                curr_sr.fit(X_train, y_train)
                fold_mse = mean_squared_error(curr_sr.predict(X_val), y_val)
                folds.append((curr_sr, fold_mse))

            sr = sorted(folds, key=lambda x: x[1])[0][0]
            self.sr = sr

        else:
            curr_sr = self.get_new_sr(**kwargs)
            curr_sr.set_params(
                binary_operators=binary_ops,  # type:ignore
                unary_operators=[*unary_ops, *[x['julia'] for x in extra_unary.values()]],  # type:ignore
                extra_sympy_mappings={x[0]: x[1]['sympy'] for x in extra_unary.items()},  # type:ignore

                elementwise_loss=custom_loss if custom_loss else 'L2DistLoss()',  # type:ignore

                constraints={'^': (-1, 1)} | constraints,
                random_state=self.random_state,  # type:ignore
                deterministic=True,  # type:ignore
                parallelism='serial'  # type:ignore
            )
            curr_sr.fit(self.X, self.y)
            self.sr = curr_sr

    @property
    def get_eq(self) -> sp.Expr:
        return self.sr.get_best()['sympy_format']

    @property
    def get_model(self) -> PySRRegressor:
        return self.sr
