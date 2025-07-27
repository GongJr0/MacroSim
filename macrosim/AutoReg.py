import pandas as pd
import numpy as np
import sympy as sp  # type: ignore
import pickle
from subprocess import Popen, PIPE
import os

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error  # type: ignore
from sklearn.utils import shuffle  # type: ignore

from typing import cast, Optional
import warnings

from pysr import PySRRegressor, TemplateExpressionSpec, ExpressionSpec  # type: ignore

from macrosim.stats.StatsTypes import DATA, LAG

os.environ["PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION"] = "no"

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


class _BatchedPredictor:
    def __init__(self,
                 models: dict[int, PySRRegressor],
                 target: str,
                 target_lags: int,
                 causal_lag_positions: dict[str, int],
                 fit_eval: Optional[dict[str, float | int]] = None):
        self._models = models
        self._target = target
        self._fit_eval = fit_eval or {}
        self._target_lags = target_lags
        self._causal_lag_positions = causal_lag_positions

    def __repr__(self):
        return self.fit_eval.__repr__()

    def __str__(self):
        return self.fit_eval.__str__()

    def reconstruct_lags(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame), "X must be a DataFrame with a valid index."
        assert self._target in X.columns, f"Target '{self._target}' not found in DataFrame."

        target = X[self._target].copy()
        target_lags = [target.shift(lag) for lag in range(1, self._target_lags)]  # Current target is lag-1 when predicting t+1

        target_df = pd.concat([target, *target_lags], axis=1)
        target_df.columns = [f"{self._target}_L{lag}" for lag in range(1, self._target_lags + 1)]  # Feature name mathing

        causal_lags = pd.concat([X[causal].shift(lag-1) for causal, lag in self._causal_lag_positions.items()], axis=1)
        causal_lags.columns = [f"{causal}_L{lag}" for causal, lag in self._causal_lag_positions.items()]

        X = pd.concat([target_df, causal_lags], axis=1)
        X = X.dropna(how='any')  # Drop rows with any NaN values
        return X

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = self.reconstruct_lags(X)

        preds = sum([model.predict(X.to_numpy()) for model in self.models.values()]) / len(self.models)
        return pd.Series(preds, index=X.index, name='prediction')

    @property
    def equations(self) -> list[pd.DataFrame]:
        return [m.get_best() for m in self.models.values()]

    @property
    def models(self) -> dict[int, PySRRegressor]:
        return self._models

    @property
    def fit_eval(self) -> dict[str, float | int]:
        return self._fit_eval

    def __getitem__(self, key: int) -> PySRRegressor:
        return self.models[key]

    def __len__(self) -> int:
        return len(self.models)


class AutoReg:
    def __init__(self, df: pd.DataFrame, target: str):
        assert df.shape[1]>1, "DataFrame must contain at least one feature column and the target column."
        assert target in df.columns, f"Target column '{target}' not found in DataFrame."

        self.df: pd.DataFrame = df
        self.target: str = target

        self.freq: int = AutoReg.get_freq(df)
        self.n_lags = cast(LAG, 2 * self.freq)  # 2m heuristic, can be adjusted based on domain knowledge

        self.causal_lag_positions: dict[str, int] = {}  # Positions of causal lags in the DataFrame, populated at process_data time

        self.X, self.y = self.process_data()

        self.model: Optional[PySRRegressor] = None  # Populated when fitting non-batched models
        self.batched_model: Optional[_BatchedPredictor] = None  # Populated when fitting batched models

    @staticmethod
    def get_freq(df: DATA) -> int:
        assert isinstance(df.index, pd.DatetimeIndex), "DatetimeIndex is mandatory for frequency inference."
        assert isinstance((freq_full := pd.infer_freq(df.index)), str), ("Frequency inference failed. Ensure the index is"
                                                                         " a DatetimeIndex with a consistent frequency.")

        freq_base = freq_full.split('-')[0]  # Get the base frequency (e.g., 'D', 'M', 'Q', etc.)
        freq = FREQ_TO_PERIODS_PER_YEAR[freq_base]
        return freq

    def get_target_lags(self) -> pd.DataFrame:

        target_frame = self.df[self.target].to_frame()
        for lag in range(1, self.n_lags + 1):
            target_frame[f"{self.target}_L{lag}"] = target_frame[self.target].shift(lag)
        target_frame = target_frame.drop(columns=[self.target])
        return target_frame

    def get_causal_lags(self) -> pd.DataFrame | None:
        if (causal_dict := self.df.causality.tests[self.target]) is None:
            return pd.DataFrame()

        lags: list[pd.DataFrame] = []
        for col, res in causal_dict.items():
            if not res:
                continue

            period = sorted(res, key=lambda x: x[1])[0][0]  # Get the lag with the lowest p-value
            self.causal_lag_positions[col] = period  # Store the position of the causal lag
            lags.append(self.df[col].shift(period).to_frame(name=f"{col}_L{period}"))
        causal_frame = pd.concat(lags, axis=1) if lags else pd.DataFrame()
        return causal_frame

    def process_data(self) -> tuple[pd.DataFrame, pd.Series]:
        y = self.df[self.target].copy()

        target_lags = self.get_target_lags()
        causal_lags = self.get_causal_lags()

        if causal_lags is None:
            target_lags = target_lags.dropna(how='any')
            return target_lags, y.loc[target_lags.index]

        X = pd.concat([target_lags, causal_lags], axis=1)
        X = X.dropna(how='any')
        return X, y.loc[X.index]

    def get_expr(self) -> TemplateExpressionSpec:
        var_set = [col for col in self.X.columns if col.startswith(self.target)]
        params = {"w": len(var_set)}

        sub_expr = " + ".join([f"w[{n+1}] * {col}" for n, col in enumerate(var_set)])  # n+1 to match julia's 1-based indexing
        return TemplateExpressionSpec(
            expressions=["f"],
            variable_names=var_set,
            parameters=params,
            combine=f"f({sub_expr})"
        )

    def model_config(self, spec: Optional[TemplateExpressionSpec] = None) -> PySRRegressor:
        size = len(self.X.columns) // 2
        spec = spec or ExpressionSpec()
        model = PySRRegressor(
            model_selection='best',

            niterations=20,
            maxsize=size if size > 7 else 7,

            expression_spec=spec,

            binary_operators=['+', '-', '*', '/', 'pow'],
            unary_operators=['exp', 'log', 'sqrt'],#, 'sin', 'cos', 'tan'],

            constraints={
                # 'sin': 2,
                # 'cos': 2,
                # 'tan': 2,

                'exp': 2,
                'log': 2,
                'sqrt': 3,
                'pow': (-1, 2)
            },
            nested_constraints={#'sin': {'cos': 0, 'sin': 2},
            #                     'cos': {'sin': 0, 'cos': 2},
            #                     'tan': {'sin': 1, 'cos': 1},
                                'exp': {'log': 0},
                                'log': {'exp': 0},
                                },

            elementwise_loss='L2DistLoss()',

            temp_equation_file=True,
            verbosity=1,
        )
        return model

    def fit(self, template_spec: bool = True) -> tuple[PySRRegressor, dict[str, float | int]]:
        X, y = self.X, self.y
        spec = self.get_expr() if template_spec else None

        model = self.model_config(spec)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
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
        self.model = model
        return model, fit_eval

    def shuffle_batch(self, n: int) -> tuple[enumerate, tuple[pd.DataFrame, pd.Series]]:
        X, y = self.X, self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            X, y = shuffle(X_train, y_train, random_state=42)

            X_batched = np.array_split(X, n)
            y_batched = np.array_split(y, n)

        return enumerate(zip(X_batched, y_batched)), (X_test, y_test)

    def batch_fit(self, n: int, template_spec: bool=True) -> _BatchedPredictor:
        train, test = self.shuffle_batch(n)
        spec = self.get_expr() if template_spec else None

        fitted: dict[int, PySRRegressor] = {}
        for i, data in train:
            X, y = data
            model = self.model_config(spec)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model.fit(X, y)
            fitted[i] = model

        X_test, y_test = test
        preds = sum([m.predict(X_test) for m in fitted.values()]) / len(fitted)

        fit_eval = {
            'R^2': r2_score(y_test, preds),
            'MAE': mean_absolute_error(y_test, preds),
            'MAPE': mean_absolute_percentage_error(y_test, preds),
            'MSE': mean_squared_error(y_test, preds),
            'n_features': len(X_test.columns),
            'n_samples_per_batch': (len(self.df)*0.8) // n,
        }
        self.batched_model = _BatchedPredictor(fitted,
                                               self.target,
                                               self.n_lags,
                                               self.causal_lag_positions,
                                               fit_eval)
        return self.batched_model

    def _batch_fit_parallel(self, n: int, template_spec: bool = True) -> _BatchedPredictor:
        train, test = self.shuffle_batch(n)

        batches: list[tuple[int, tuple[pd.DataFrame, DATA]]] = list(train)
        dummy = batches[0][1]
        X_dummy = dummy[0].iloc[0]
        y_dummy = dummy[1].iloc[0]

        if not os.path.exists('./batch/data/'):
            os.makedirs('./batch/data/', exist_ok=True)

        for no, data in batches:
            with open(f'./batch/data/data_{no}.pkl', 'wb') as f:
                pickle.dump(data, f)  # type: ignore

        procs = []
        for no, _ in batches:
            proc = Popen(['python', 'macrosim/_batch_fit.py', f'./batch/data/data_{no}.pkl', str(no), str(template_spec)],
                         stdout=PIPE,
                         stderr=PIPE)
            procs.append(proc)

        for p in procs:
            stdout, stderr = p.communicate()
            if p.returncode != 0:
                print(f"[STDOUT]: \n{stdout.decode('utf-8')}")
                raise RuntimeError(f"[ERROR] Fit failed for batch {p.args[3]}:\n{stderr.decode('utf-8')}")
            else:
                print(f"[INFO] Fit completed for batch {p.args[3]}:\n{stdout.decode('utf-8')}")

        models = {no: self.model_config(self.get_expr() if template_spec else None) for no, _ in batches}
        for no, model in models.items():
            with open(f'./batch/eq/eq_{no}.pkl', 'rb') as f:
                best = pickle.load(f)
                if isinstance(model.expression_spec, TemplateExpressionSpec):
                    varnames = model.expression_spec.variable_names
                    eq = best['lambda_format']
                    eq_lambda = sp.lambdify(varnames, eq, modules='numpy')

            model.fit(X=X_dummy.to_frame().T, y=np.array([y_dummy]))

            best = model.get_best().to_frame().copy()
            best['equation'] = str(eq)
            best['labmda_format'] = eq_lambda

            model.equations_ = best.T
            print(model.equations_['loss'])

        X_test, y_test = test
        preds = sum([m.predict(X_test) for m in models.values()]) / len(models)

        fit_eval = {
            'R^2': r2_score(y_test, preds),
            'MAE': mean_absolute_error(y_test, preds),
            'MAPE': mean_absolute_percentage_error(y_test, preds),
            'MSE': mean_squared_error(y_test, preds),
            'n_features': len(X_test.columns),
            'n_samples_per_batch': (len(self.df) * 0.8) // n,
        }
        self.batched_model = _BatchedPredictor(
            models,
            self.target,
            self.n_lags,
            self.causal_lag_positions,
            fit_eval=fit_eval
        )
        return self.batched_model