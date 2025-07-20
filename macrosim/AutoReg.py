import pandas as pd
import sympy as sp  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error  # type: ignore
from typing import cast, Optional

from pysr import PySRRegressor, TemplateExpressionSpec, ExpressionSpec  # type: ignore

from .StatsTypes import DATA, LAG

FREQ_TO_PERIODS_PER_YEAR = {
    "D": 365,
    "B": 252,  # Assumed 252, can change year to year
    "W": 52,
    "M": 12,
    "MS": 12,
    "Q": 4,
    "QS": 4,
    "A": 1,
    "AS": 1,
    "Y": 1,
    "H": 24 * 365,
    "T": 60 * 24 * 365,
    "min": 60 * 24 * 365,
    "S": 60 * 60 * 24 * 365,
}


class AutoReg:
    def __init__(self):
        # Static Class
        ...

    @staticmethod
    def get_freq(df: DATA) -> int:
        assert isinstance(df.index, pd.DatetimeIndex), "DatetimeIndex is mandatory for frequency inference."
        assert isinstance((freq_full := pd.infer_freq(df.index)), str), "Frequency inference failed. Ensure the index is a DatetimeIndex with a consistent frequency."

        freq_base = freq_full.split('-')[0]  # Get the base frequency (e.g., 'D', 'M', 'Q', etc.)
        freq = FREQ_TO_PERIODS_PER_YEAR[freq_base]
        return freq

    @staticmethod
    def get_target_lags(df: DATA, target: str) -> pd.DataFrame:
        n_lags: LAG = cast(LAG, 2 * AutoReg.get_freq(df))  # 2m heuristic (2 full cycles)

        target_frame = df[target].to_frame()
        for lag in range(1, n_lags + 1):
            target_frame[f"{target}_L{lag}"] = target_frame[target].shift(lag)
        target_frame = target_frame.drop(columns=[target])
        return target_frame

    @staticmethod
    def get_causal_lags(df: DATA, target: str) -> pd.DataFrame | None:
        if (causal_dict := df.causality.tests[target]) is None:
            return None

        lags: list[pd.DataFrame] = []
        for col, res in causal_dict.items():
            if not res:
                continue

            period = sorted(res, key=lambda x: x[1])[0][0]  # Get the lag with the lowest p-value
            lags.append(df[col].shift(period).to_frame(name=f"{col}_L{period}"))
        causal_frame = pd.concat(lags, axis=1)
        return causal_frame

    @staticmethod
    def process_data(df: DATA, target: str) -> tuple[pd.DataFrame, pd.Series]:
        assert target in df.columns, f"Target column '{target}' not found in DataFrame."

        y = df[target].copy()

        target_lags = AutoReg.get_target_lags(df, target)
        causal_lags = AutoReg.get_causal_lags(df, target)

        if causal_lags is None:
            target_lags = target_lags.dropna(how='any')
            return target_lags, y.loc[target_lags.index]

        X = pd.concat([target_lags, causal_lags], axis=1)
        X = X.dropna(how='any')
        return X, y.loc[X.index]

    @staticmethod
    def get_expr(X: pd.DataFrame, target: str) -> TemplateExpressionSpec:
        var_set = [col for col in X.columns if col.startswith(target)]
        params = {"w": len(var_set)}

        sub_expr = " + ".join([f"w[{n+1}] * {col}" for n, col in enumerate(var_set)])  # n+1 to match julia's 1-based indexing
        return TemplateExpressionSpec(
            expressions=["f"],
            variable_names=var_set,
            parameters=params,
            combine=f"f({sub_expr})"
        )

    @staticmethod
    def model_config(X: pd.DataFrame, spec: Optional[TemplateExpressionSpec] = None) -> PySRRegressor:
        size = len(X.columns) * 2
        spec = spec or ExpressionSpec()
        model = PySRRegressor(
            model_selection='best',

            niterations=200,
            maxsize=size,

            expression_spec=spec,

            binary_operators=['+', '-', '*', '/', 'pow'],
            unary_operators=['exp', 'log', 'sqrt', 'sin', 'cos', 'tan'],

            constraints={
                'sin': 2,
                'cos': 2,
                'tan': 2,

                'exp': 2,
                'log': 2,
                'sqrt': 3,
                'pow': (-1, 2)
            },
            nested_constraints={'sin': {'cos': 0, 'sin': 2},
                                'cos': {'sin': 0, 'cos': 2},
                                'tan': {'sin': 1, 'cos': 1},
                                'exp': {'log': 0},
                                'log': {'exp': 0},
                                },

            elementwise_loss='L2DistLoss()',

            temp_equation_file=True,
        )
        return model

    @staticmethod
    def fit(df: pd.DataFrame, target: str) -> tuple[PySRRegressor, dict[str, float | int]]:
        X, y = AutoReg.process_data(df, target)
        spec = AutoReg.get_expr(X, target)

        model = AutoReg.model_config(df, spec)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        fit_eval = {
            'R^2': r2_score(y_test, pred),
            'MAE': mean_absolute_error(y_test, pred),
            'MAPE': mean_absolute_percentage_error(y_test, pred),
            'MSE': mean_squared_error(y_test, pred),
            'n_features': len(X_train.columns),
            'n_samples': len(X_train)
        }
        return model, fit_eval
