import pandas as pd
import numpy as np
import sympy as sp  # type: ignore
import pickle
from subprocess import run, PIPE, Popen
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import psutil
import os
import datetime as dt
import builtins

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error  # type: ignore
from sklearn.utils import shuffle  # type: ignore

from typing import cast, Optional
import warnings

from pysr import PySRRegressor, TemplateExpressionSpec, ExpressionSpec, TensorBoardLoggerSpec  # type: ignore

from macrosim.stats.StatsTypes import DATA, LAG, FREQ_TO_PERIODS_PER_YEAR


def run_batch_process(args):
    no, template_spec = args
    cmd = ['python', 'macrosim/_batch_fit.py', f'./batch/data/data_{no}.pkl', str(no), str(template_spec)]

    proc = run(cmd, stdout=PIPE, stderr=PIPE)
    return no, proc.stdout.decode(), proc.stderr.decode()


def ssqrt(x: float) -> float:
    return np.sign(x) * np.sqrt(np.abs(x))


if 'ssqrt' not in builtins.__dict__:
    builtins.ssqrt = ssqrt  # type: ignore

os.environ["PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION"] = "no"


class _BatchedPredictor:
    def __init__(self,
                 models: dict[int, PySRRegressor],
                 target: str,
                 target_lags: int,
                 causal_lag_positions: dict[str, int],
                 fit_eval: Optional[dict[str, float | int]] = None):

        if isinstance(models[0], PySRRegressor):
            eqs = {i: model.get_best()['lambda_format'] for i, model in models.items()}
        else:
            eqs = models
        self._models = models
        self._eqs = eqs
        self._target = target
        self._fit_eval = fit_eval or {}
        self._target_lags = target_lags
        self._causal_lag_positions = causal_lag_positions

    def __repr__(self):
        return self.fit_eval.__repr__()

    def __str__(self):
        return self.fit_eval.__str__()

    def reconstruct_lags(self, X: pd.DataFrame, t_range: pd.DatetimeIndex) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame), "X must be a DataFrame with a valid index."
        assert self._target in X.columns, f"Target '{self._target}' not found in DataFrame."

        target = X[self._target].copy()
        target_lags = [target.shift(lag-1) for lag in range(1, self._target_lags)]  # Current target is lag_1 when predicting t+1

        target_df = pd.concat([target, *target_lags], axis=1)
        target_df.columns = [f"{self._target}_L{lag}" for lag in range(1, self._target_lags + 1)]  # Feature name mathing

        causal_lags = pd.concat([X[causal].shift(lag) for causal, lag in self._causal_lag_positions.items()], axis=1)
        causal_lags.columns = [f"{causal}_L{lag}" for causal, lag in self._causal_lag_positions.items()]

        X = pd.concat([target_df, causal_lags], axis=1)
        X = X.dropna(how='any')  # Drop rows with any NaN values
        X = X.reindex(t_range)
        return X

    def predict(self, X: pd.DataFrame, t_range: pd.DatetimeIndex) -> pd.Series:
        X = self.reconstruct_lags(X, t_range)

        preds = sum([eq(*X.values.T) for eq in self.eqs.values()]) / len(self.eqs)
        return pd.Series(preds, index=X.index, name='prediction')

    @property
    def equations(self) -> list[pd.DataFrame]:
        return [m.get_best() for m in self.models.values()]

    @property
    def models(self) -> dict[int, PySRRegressor]:
        return self._models

    @property
    def eqs(self) -> dict[int, sp.Lambda]:
        return self._eqs

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
        self.n_lags = cast(LAG, 2*self.freq)  # 2m heuristic, can be adjusted based on domain knowledge

        self.causal_lag_positions: dict[str, int] = {}  # Positions of causal lags in the DataFrame, populated at process_data time

        self.X, self.y = self.process_data()

        self.model: PySRRegressor = cast(PySRRegressor, None)  # Populated when fitting non-batched models
        self.batched_model: _BatchedPredictor = cast(_BatchedPredictor, None)  # Populated when fitting batched models

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

    def model_config(self,
                     spec: Optional[TemplateExpressionSpec] = None,
                     log: bool = True,
                     logdir_child: Optional[str] = None) -> PySRRegressor:

        # ==== Custom Operators ====
        def safe_sqrt(x) -> float:
            """This function has been used as a sign safe root replacement in many academic settings.
            For a specific example matching the domain of AutoReg, refer to:

            Teräsvirta, T. (1994). Specification, Estimation, and Evaluation of Smooth Transition Autoregressive Models.
            Journal of the American Statistical Association.

            builtins alias: ssqrt
            """
            return np.sign(x) * np.sqrt(np.abs(x))
        # ===========================

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
            unary_operators=['exp', 'log', 'ssqrt(x) = sign(x)*sqrt(abs(x))','sin', 'cos', 'tan'],
            extra_sympy_mappings={'ssqrt': safe_sqrt},

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

    def batch_fit(self, n: int, *, template_spec: bool = True, parallel: bool = True) -> _BatchedPredictor:
        if parallel:
            return self._batch_fit_parallel(n, template_spec)

        train, test = self.shuffle_batch(n)
        spec = self.get_expr() if template_spec else None
        log_dir = f"{self.target}_{dt.datetime.now().strftime('%b-%d_%H-%M')}"

        fitted: dict[int, PySRRegressor] = {}
        for i, data in train:
            X, y = data
            model = self.model_config(spec, logdir_child=log_dir + f"/{i}")

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
        def monitor_process(p, batch_id):
            stdout, stderr = p.communicate()
            if p.returncode != 0:
                print(f"[ERROR] Batch {batch_id} failed:\n{stderr.decode()}")
            else:
                print(f"[INFO] Batch {batch_id} completed:\n{stdout.decode()}")

        train, test = self.shuffle_batch(n)

        batches: list[tuple[int, tuple[pd.DataFrame, DATA]]] = list(train)
        dummy = batches[0][1]

        if not os.path.exists('./batch/data/'):
            os.makedirs('./batch/data/', exist_ok=True)

        for no, data in batches:
            with open(f'./batch/data/data_{no}.pkl', 'wb') as f:
                pickle.dump(data, f)  # type: ignore

        procs = []
        threads = []
        for no, _ in batches:
            proc = Popen(['python', 'macrosim/_batch_fit.py', f'./batch/data/data_{no}.pkl', str(no), str(template_spec)],
                         stdout=PIPE,
                         stderr=PIPE)
            procs.append(proc)

            t = threading.Thread(target=monitor_process, args=(proc, no))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()


        # task_args = [(no, template_spec) for no, _ in batches]
        #
        # cpus = psutil.cpu_count(logical=False)
        # with ProcessPoolExecutor(max_workers=cpus) as executor:
        #     futures = {executor.submit(run_batch_process, args): args for args in task_args}
        #
        #     for future in as_completed(futures):
        #         no, out, err = future.result()
        #         print(f"[{no}] STDOUT:\n{out}")
        #         print(f"[{no}] STDERR:\n{err}")

        eqs = {}
        models = {no: self.model_config(self.get_expr() if template_spec else None) for no, _ in batches}
        for no, model in models.items():
            with open(f'./batch/eq/eq_{no}.pkl', 'rb') as f:  # type: ignore
                features, best = pickle.load(f)
                if isinstance(model.expression_spec, TemplateExpressionSpec):
                    var_symbols = sp.symbols(features)
                    eq = best['lambda_format']
                    eq_lambda = sp.lambdify(var_symbols, eq, modules='numpy')
                    eqs[no] = eq_lambda
            os.remove(f'./batch/eq/eq_{no}.pkl')

        batch_model = _BatchedPredictor(eqs,
                                        self.target,
                                        self.n_lags,
                                        self.causal_lag_positions)
        X_test, y_test = test
        X_test = X_test.reindex(columns=features)
        y_test = y_test.reindex(X_test.index)

        preds = batch_model.predict(self.df, cast(pd.DatetimeIndex, X_test.index))

        fit_eval = {
            'R^2': r2_score(y_test, preds),
            'MAE': mean_absolute_error(y_test, preds),
            'MAPE': mean_absolute_percentage_error(y_test, preds),
            'MSE': mean_squared_error(y_test, preds),
            'n_features': len(X_test.columns),
            'n_samples_per_batch': (len(self.df) * 0.8) // n,
        }
        batch_model._fit_eval = fit_eval
        self.batched_model = batch_model

        return self.batched_model
