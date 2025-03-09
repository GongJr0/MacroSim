from typing import Optional, Any, Literal

from jedi.inference.gradual.typing import Callable
from pysr import PySRRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.neighbors import LocalOutlierFactor

import pandas as pd
import numpy as np

from math import e

VALID_BINARY_OPS = list[Literal['+', '-', '*', '/', '^']]
FULL_BINARY_OPS: VALID_BINARY_OPS = ['+', '-', '*', '/', '^']

DEFAULT_UNARY_OPS: list = ['exp', 'log', 'sqrt', 'sin', 'cos']

class EqSearch:
    """
    Automated symbolic search engine using PySR's cutting-edge SymbolicRegressor model.
    """

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.DataFrame, # y.shape = (-1, 1)
                 random_state: int = 0) -> None:
        self.extra_unary: dict[str, dict[str, Any]] = {
            'inv': {
                'julia': 'inv(x)=1/x',
                'sympy': lambda x: 1/x
            }
        }

        self.X = X
        self.y = y

        self.random_state = random_state

        self.model = PySRRegressor()

        self.distilled = None

        self.eq: Optional[Callable] = None

    def distil_split(self, test_size: float = 0.2,
                     grid_search: bool = False, gs_params: Optional[dict[str, Any]] = ...) -> pd.DataFrame:
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

        print(f"RandomForest Score at Distillation: {rf.score(X_test, y_test)}")

        distilled_y = rf.predict(self.X)

        self.distilled = pd.DataFrame(distilled_y, index=self.X.index)

    def search(self, binary_ops: VALID_BINARY_OPS = FULL_BINARY_OPS, unary_ops = DEFAULT_UNARY_OPS,
               extra_unary_ops: dict[str, dict[str, Any]] = {},
               custom_loss: Optional[str] = None):
        assert self.distilled.shape == self.y.shape, "Run self.distil_split() before symbolizing."

        extra_unary = self.extra_unary | extra_unary_ops

        sr = PySRRegressor(model_selection='accuracy',  # Do not consider complexity at selection

                           maxsize=30,
                           niterations=300,

                           binary_operators=binary_ops,
                           unary_operators=[*unary_ops, *[x['julia'] for x in extra_unary.values()]],
                           extra_sympy_mappings={x[0]: x[1]['sympy'] for x in extra_unary.items()}, # type: ignore

                           elementwise_loss=custom_loss if custom_loss else 'L2DistLoss()',

                           verbosity=1,
                           progress=False,
                           temp_equation_file=True
                           )
        sr.fit(self.X, self.distilled)

        print(sr.get_best())

        self.eq: Callable = sr.get_best()['lambda_format']
