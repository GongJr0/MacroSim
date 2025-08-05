import sys

import pandas as pd
import numpy as np
import sympy as sp  # type: ignore
import datetime as dt

import pickle
import builtins

from subprocess import run, PIPE, Popen
import threading
import os


from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error  # type: ignore
from sklearn.utils import shuffle  # type: ignore

from abc import ABC, abstractmethod
from typing import cast, Union, Optional, Callable, Literal, Any
import warnings

from pysr import PySRRegressor, TemplateExpressionSpec, ExpressionSpec, TensorBoardLoggerSpec  # type: ignore

from macrosim.stats.StatsTypes import DATA, LAG, FREQ_TO_PERIODS_PER_YEAR, SeriesInfo

def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    print(f"{category.__name__}: {message}", file=sys.stderr)
warnings.showwarning = custom_showwarning

def run_batch_process(args):
    no, template_spec = args
    cmd = ['python', 'macrosim/_batch_fit.py', f'./batch/data/data_{no}.pkl', str(no), str(template_spec)]

    proc = run(cmd, stdout=PIPE, stderr=PIPE)
    return no, proc.stdout.decode(), proc.stderr.decode()


# ====== Custom Operators ======
def ssqrt(x: float) -> float:
    """This function has been used as a sign safe root replacement in many academic settings.
    For a specific example matching the domain of AutoReg, refer to:

    Teräsvirta, T. (1994). Specification, Estimation, and Evaluation of Smooth Transition Autoregressive Models.
    Journal of the American Statistical Association."""
    return np.sign(x) * np.sqrt(np.abs(x))  # type: ignore
                                            # mypy cannot determine the return of np.abs

# ===========================


EXTRA_GLOBALS: dict[str, Callable[[float], float]] = {
    'ssqrt': ssqrt
}

for n, f in EXTRA_GLOBALS.items():
    if not hasattr(builtins, n):
        setattr(builtins, n, f)  # Prevent reference before assignment

os.environ["PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION"] = "no"


class _AbstractPredictor(ABC):
    @property
    @abstractmethod
    def target(self) -> str: ...

    @property
    @abstractmethod
    def target_lags(self) -> LAG: ...

    @property
    @abstractmethod
    def causal_lag_positions(self) -> dict[str, LAG]: ...

    @property
    @abstractmethod
    def ordered_features(self) -> Union[dict[int, list[str]], list[str]]: ...

    @property
    @abstractmethod
    def fit_eval(self) -> dict[str, float | int]: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame, t_range: pd.DatetimeIndex) -> pd.Series: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target={self.target})"

    def __str__(self) -> str:
        return self.fit_eval.__str__()

    def reconstruct_lags(self, X: pd.DataFrame, t_range: pd.DatetimeIndex) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame), "X must be a DataFrame with a valid index."
        assert self.target in X.columns, f"Target '{self.target}' not found in DataFrame."

        target = X[self.target].copy()
        target_lags = [target.shift(lag - 1) for lag in
                       range(1, self.target_lags)]  # Current target is lag_1 when predicting t+1

        target_df = pd.concat([target, *target_lags], axis=1)
        target_df.columns = [f"{self.target}_L{lag}" for lag in
                             range(1, self.target_lags + 1)]  # Feature name mathing
        if self.causal_lag_positions:
            causal_lags = pd.concat([X[causal].shift(lag) for causal, lag in self.causal_lag_positions.items()], axis=1)
            causal_lags.columns = [f"{causal}_L{lag}" for causal, lag in self.causal_lag_positions.items()]
        else:
            causal_lags = pd.DataFrame()

        X = pd.concat([target_df, causal_lags], axis=1)
        X = X.dropna(how='any')  # Drop rows with any NaN values
        X = X.reindex(t_range)
        return X


class _Predictor(_AbstractPredictor):
    def __init__(self,
                 model: PySRRegressor,
                 target: str,
                 target_lags: LAG,
                 causal_lag_positions: dict[str, LAG],
                 fit_eval: Optional[dict[str, float | int]] = None):
        self._model: PySRRegressor = model
        self._target: str = target
        self._target_lags: LAG = target_lags
        self._causal_lag_positions: dict[str, LAG] = causal_lag_positions
        self._fit_eval: dict[str, float] = fit_eval or {}

    def predict(self, X: pd.DataFrame, t_range: pd.DatetimeIndex) -> pd.Series:
        X = self.reconstruct_lags(X, t_range)

        preds = self._model.predict(X)
        return pd.Series(preds, index=X.index, name='prediction')

    @property
    def target(self) -> str:
        return self._target

    @property
    def target_lags(self) -> LAG:
        return self._target_lags

    @property
    def causal_lag_positions(self) -> dict[str, LAG]:
        return self._causal_lag_positions

    @property
    def ordered_features(self) -> list[str]:
        return self._model.feature_names_in_.tolist()

    @property
    def model(self) -> PySRRegressor:
        return self._model

    @property
    def fit_eval(self) -> dict[str, float | int]:
        rounded =  {k: round(v, 4) if isinstance(v, float) else v for k, v in self._fit_eval.items()}
        return rounded


class _BatchedPredictor(_AbstractPredictor):
    def __init__(self,
                 eqs: dict[int, Callable],
                 ordered_features: dict[int, list[str]],
                 target: str,
                 target_lags: LAG,
                 causal_lag_positions: dict[str, LAG],
                 fit_eval: Optional[dict[str, float | int]] = None):

        self._eqs: dict[int, Callable] = eqs
        self._ordered_features: dict[int, list[str]] = ordered_features
        self._target: str = target
        self._fit_eval: dict[str, float] = fit_eval or {}
        self._target_lags: LAG = target_lags
        self._causal_lag_positions: dict[str, LAG] = causal_lag_positions

    def predict(self, X: pd.DataFrame, t_range: pd.DatetimeIndex) -> pd.Series:
        X = self.reconstruct_lags(X, t_range)

        preds = sum([eq(*X[self.ordered_features[i]].values.T) for i, eq in self.eqs.items()]) / len(self.eqs)
        return pd.Series(preds, index=X.index, name='prediction')

    @property
    def target(self) -> str:
        return self._target

    @property
    def target_lags(self) -> LAG:
        return self._target_lags

    @property
    def causal_lag_positions(self) -> dict[str, LAG]:
        return self._causal_lag_positions

    @property
    def ordered_features(self) -> dict[int, list[str]]:
        return self._ordered_features

    @property
    def eqs(self) -> dict[int, Callable]:
        return self._eqs

    @property
    def fit_eval(self) -> dict[str, float | int]:
        rounded =  {k: round(v, 4) if isinstance(v, float) else v for k, v in self._fit_eval.items()}
        return rounded

    def __getitem__(self, key: int) -> Callable:
        return self.eqs[key]

    def __len__(self) -> int:
        return len(self.eqs)


class AutoReg:
    def __init__(self, df: pd.DataFrame, target: str, assert_stationarity: bool = False):
        assert df.shape[1] > 1, "DataFrame must contain at least one feature column and the target column."
        assert target in df.columns, f"Target column '{target}' not found in DataFrame."

        if assert_stationarity:
            warnings.warn("Enable assert_stationary with caution! MacroSim uses a 20% significance level for stationarity tests due "
                          "to macroeconomic variables often being non-stationary under strict terms. "
                          "Even with the relaxed significance level, you should expect most variables to be non-stationary.",
                          category=UserWarning)

        self.df: pd.DataFrame = df
        self.target: str = target

        self.freq: int = AutoReg.get_freq(df)
        self.n_lags: LAG = cast(LAG, 2 * self.freq)  # 2m heuristic, can be adjusted based on domain knowledge

        self.assert_stationarity: bool = assert_stationarity
        self.causal_lag_positions: dict[
            str, LAG] = {}  # Positions of causal lags in the DataFrame, populated at process_data time

        self.X, self.y = self.process_data()

        self.model: PySRRegressor = cast(PySRRegressor, None)  # Populated when fitting non-batched models
        self.batched_model: _BatchedPredictor = cast(_BatchedPredictor, None)  # Populated when fitting batched models

        self.custom_params: dict[str, Any] | None = None  # Custom parameters for model configuration, populated when set_params is called

# Data Prep
    @staticmethod
    def get_freq(df: DATA) -> int:
        assert isinstance(df.index, pd.DatetimeIndex), "DatetimeIndex is mandatory for frequency inference."
        assert isinstance((freq_full := pd.infer_freq(df.index)), str), (
            "Frequency inference failed. Ensure the index is"
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

    def check_stationarity(self, col) -> bool:
        return self.df.stationarity.is_stationary[col] == SeriesInfo.STATIONARY

    def get_causal_lags(self) -> pd.DataFrame | None:
        if (causal_dict := self.df.causality.tests[self.target]) is None:
            return pd.DataFrame()

        lags: list[pd.DataFrame] = []
        for col, res in causal_dict.items():
            if not res:
                continue
            if self.assert_stationarity and (not self.check_stationarity(col)):
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

    def shuffle_batch(self, n: int) -> tuple[enumerate, tuple[pd.DataFrame, pd.Series]]:
        X, y = self.X, self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            X, y = shuffle(X_train, y_train, random_state=42)

            X_batched = np.array_split(X, n)
            y_batched = np.array_split(y, n)

        return enumerate(zip(X_batched, y_batched)), (X_test, y_test)

    # Model Config
    def get_expr(self) -> TemplateExpressionSpec:
        var_set = [col for col in self.X.columns if col.startswith(self.target)]
        params = {"w": len(var_set)}

        sub_expr = " + ".join(
            [f"w[{n + 1}] * {col}" for n, col in enumerate(var_set)])  # n+1 to match julia's 1-based indexing
        return TemplateExpressionSpec(
            expressions=["f"],
            variable_names=var_set,
            parameters=params,
            combine=f"f({sub_expr})"
        )

    def set_params(self, **kwargs) -> None:
        assert set(kwargs.keys()) <= self._PARAM_SET, (f"Invalid Parameters Passed: {set(kwargs.keys()) - self._PARAM_SET}\n\n"
                                                       f"Available Parameters: {self._PARAM_SET}")
        self.custom_params = kwargs

    def model_config(self,
                     spec: Optional[TemplateExpressionSpec] = None,
                     log: bool = True,
                     logdir_child: Optional[str] = None) -> PySRRegressor:

        size = len(self.X.columns) * 2
        spec = spec or ExpressionSpec()
        if log:
            dir_name = f"logs/{logdir_child}" if logdir_child else f"logs/{self.target}_{dt.datetime.now().strftime('%b-%d_%H-%M-%S')}"
            logger = TensorBoardLoggerSpec(
                log_dir=dir_name,
                log_interval=5,
                overwrite=False,
            )
        else:
            logger = None

        model = PySRRegressor(
            model_selection='best',

            niterations=200,
            maxsize=size if size > 7 else 7,

            expression_spec=spec,

            binary_operators=['+', '-', '*', '/', 'pow'],
            unary_operators=['exp', 'log', 'ssqrt(x) = sign(x)*sqrt(abs(x))', 'sin', 'cos', 'tan'],
            extra_sympy_mappings={'ssqrt': ssqrt},

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
            verbosity=1,
            logger_spec=logger,
        )
        if self.custom_params:
            model.set_params(**self.custom_params)
            
        return model

    # Fit Methods
    def fit(self,
            template_spec: bool = True) -> tuple[_Predictor, dict[str, float | int]]:
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
        model = _Predictor(model, self.target, self.n_lags, self.causal_lag_positions, fit_eval)

        self.model = model
        return model, fit_eval

    def batch_fit(self,
                  n: int,
                  *,
                  template_spec: bool = True,
                  parallel: bool = True) -> _BatchedPredictor:
        if parallel:
            return self._batch_fit_parallel(n, template_spec)

        train, test = self.shuffle_batch(n)
        spec = self.get_expr() if template_spec else None
        log_dir = f"{self.target}_{dt.datetime.now().strftime('%b-%d_%H-%M')}"

        eqs: dict[int, Callable] = {}
        features_ordered = {}
        for i, data in train:
            X, y = data
            model = self.model_config(spec, logdir_child=log_dir + f"/{i}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model.fit(X, y)
            eqs[i] = model.get_best()['lambda_format']
            features_ordered[i] = model.feature_names_in_
        batched_model = _BatchedPredictor(eqs,
                                          features_ordered,
                                          self.target,
                                          self.n_lags,
                                          self.causal_lag_positions)

        X_test, y_test = test
        preds = batched_model.predict(self.df, cast(pd.DatetimeIndex, X_test.index))

        fit_eval = {
            'R^2': r2_score(y_test, preds),
            'MAE': mean_absolute_error(y_test, preds),
            'MAPE': mean_absolute_percentage_error(y_test, preds),
            'MSE': mean_squared_error(y_test, preds),
            'n_features': len(X_test.columns),
            'n_samples_per_batch': (len(self.df) * 0.8) // n,
        }
        batched_model._fit_eval = fit_eval
        self.batched_model = batched_model

        return batched_model

    def _batch_fit_parallel(self,
                            n: int,
                            template_spec: bool = True) -> _BatchedPredictor:

        def monitor_process(p, batch_id):
            stdout, stderr = p.communicate()
            if p.returncode != 0:
                print(f"[ERROR] Batch {batch_id} failed:\n{stderr.decode()}")
            else:
                print(f"[INFO] Batch {batch_id} completed:\n{stdout.decode()}")

        train, test = self.shuffle_batch(n)

        batches: list[tuple[int, tuple[pd.DataFrame, DATA]]] = list(train)

        if not os.path.exists('./batch/data/'):
            os.makedirs('./batch/data/', exist_ok=True)

        for no, data in batches:
            with open(f'./batch/data/data_{no}.pkl', 'wb') as f:
                pickle.dump(data, f)  # type: ignore

        procs = []
        threads = []
        for no, _ in batches:
            proc = Popen(
                ['python', 'macrosim/_batch_fit.py', f'./batch/data/data_{no}.pkl', str(no), str(template_spec)],
                stdout=PIPE,
                stderr=PIPE)
            procs.append(proc)

            t = threading.Thread(target=monitor_process, args=(proc, no))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        eqs = {}
        features_ordered = {}
        models = {no: self.model_config(self.get_expr() if template_spec else None) for no, _ in batches}
        for no, model in models.items():
            with open(f'./batch/eq/eq_{no}.pkl', 'rb') as f:  # type: ignore
                features, eq = pickle.load(f)
                var_symbols = sp.symbols(features)
                eq = sp.lambdify(var_symbols, eq)
                eq.__globals__.update(EXTRA_GLOBALS)

            eqs[no] = eq
            features_ordered[no] = features
        batched_model = _BatchedPredictor(eqs,
                                          features_ordered,
                                          self.target,
                                          self.n_lags,
                                          self.causal_lag_positions)
        X_test, y_test = test
        X_test = X_test.reindex(columns=features)
        y_test = y_test.reindex(X_test.index)

        preds = batched_model.predict(self.df, cast(pd.DatetimeIndex, X_test.index))

        fit_eval = {
            'R^2': r2_score(y_test, preds),
            'MAE': mean_absolute_error(y_test, preds),
            'MAPE': mean_absolute_percentage_error(y_test, preds),
            'MSE': mean_squared_error(y_test, preds),
            'n_features': len(X_test.columns),
            'n_samples_per_batch': (len(self.df) * 0.8) // n,
        }
        batched_model._fit_eval = fit_eval
        self.batched_model = batched_model

        return batched_model

    # Prediction
    def predict(self, data: pd.DataFrame, t_range: pd.DatetimeIndex, model_type: Optional[Literal['ensemble', 'single']]) -> pd.Series:
        match model_type:
            case 'ensemble':
                assert isinstance(self.batched_model, _BatchedPredictor), "Batched model not fitted."
                return self.batched_model.predict(data, t_range)

            case 'single':
                assert isinstance(self.model, _Predictor), "Batched model not fitted."
                return self.model.predict(data, t_range)

            case _:
                assert (self.batched_model is None) or (self.model is None), ("Both models are fitted. "
                                                                              "Please specify model_type.")
                if isinstance(self.model, _Predictor):
                    return self.model.predict(data, t_range)
                elif isinstance(self.batched_model, _BatchedPredictor):
                    return self.batched_model.predict(data, t_range)
                else:
                    raise ValueError("No model fitted. Please fit a model before predicting.")

    @property
    def fit_eval(self) -> dict[str, float | int] | dict[str, dict[str, float | int]]:
        if isinstance(self.model, _Predictor) and isinstance(self.batched_model, _Predictor):
            return {
                'single': self.model.fit_eval,
                'ensemble': self.batched_model.fit_eval
            }
        elif isinstance(self.model, _Predictor):
            return self.model.fit_eval

        elif isinstance(self.batched_model, _BatchedPredictor):
            return self.batched_model.fit_eval
        else:
            raise ValueError("No model fitted. Please fit a model before accessing fit_eval.")

    # Misc Properties
    @property
    def _PARAM_SET(self) -> set[str]:
        return set(
            self.model_config().get_params().keys()
        )