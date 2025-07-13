from dataclasses import asdict, make_dataclass

import sympy as sp
import pandas as pd
import numpy as np

from pysr import PySRRegressor
from sklearn.ensemble import RandomForestRegressor
from legacy.macrosim.GrowthDetector import GrowthEstimator

from typing import Callable, Union, Any, Type
from collections.abc import Generator


class SimEngine:
    """Class to iteratively simulate n periods of economic activity using a
    given symbolic equation and custom growth functions."""

    def __init__(self,
                 sr: PySRRegressor,  # Output of EqSearch
                 n_lags: int,
                 init_params: dict[str, tuple[pd.Series, GrowthEstimator]],
                 deterministic: bool = True,
                 entropy_coef: dict[str, float] = None,
                 random_state: int = 0):

        self.param_names = init_params.keys()
        self.n_lags = n_lags

        # Simloop and Growth estimator functions
        self.sr = sr
        self.growth = {
            k: init_params[k][1] for k in self.param_names
        }

        self.base_growth = {
            k: init_params[k][1] for k in self.param_names if init_params[k][1].is_base is True
        }

        self.non_base_growth = {
            k: init_params[k][1] for k in self.param_names if init_params[k][1].is_base is False
        }

        if deterministic:
            np.random.seed(random_state)

        self.param_space_df = pd.concat([init_params[param][0] for param in self.param_names], axis=1).reset_index(drop=True)
        self._out = []

        # State recording format
        self.SimState = make_dataclass('SimState', self.param_names)

        # Get init_params as the first state recording
        curr_state = self.SimState(**{
            name: self.param_space_df[name].iloc[-1] for name in self.param_names
        })
        self.state_collection = [curr_state]

        self.entropy_coef = entropy_coef if entropy_coef is not None else {
            k: 0 for k in self.param_names
        }

    @staticmethod
    def _entropy(coef) -> Generator[float, None, None]:
        yield np.random.normal(1, coef)

    def _format_lags(self):
        n_lags = self.n_lags
        data = self.param_space_df.tail(n_lags)
        if data.isna().sum().sum() > 0:
            print('NaN encountered in lagged features. Fill value used: 0')
            data = data.fillna(0)

        new_index = data.columns
        new_columns = [f"X_t{x}" for x in range(1, n_lags + 1)]
        data = data.T
        data = data.set_index(new_index)
        data.columns = new_columns
        return data

    def _simulate_step(self) -> None:

        step_lags = self._format_lags()

        base_params = {
            k: self.base_growth[k].predict(step_lags.loc[k, :].to_frame().T) for k in self.base_growth.keys()
        }

        non_base_params = {
            k: self.non_base_growth[k].predict(np.array([*list(step_lags.loc[k, :].to_frame().values), *list(base_params.values())]).reshape((1, -1)))
            for k in self.non_base_growth.keys()
        }
        unordered_params = {**base_params, **non_base_params}

        eq_params = {param: unordered_params[param] if unordered_params[param] is not None else 0 for param in self.param_names}

        model_input = np.array([var for var in eq_params.values()]).reshape(1, -1)
        out = self.sr.predict(model_input)
        self._out.extend(out)

        # Append growth to param_space
        row = pd.Series(eq_params).to_frame().T
        self.param_space_df = pd.concat([self.param_space_df, row], ignore_index=True)

        # State Record
        state = self.SimState(**eq_params)
        self.state_collection.append(state)

    def simulate(self, n_steps: int) -> pd.DataFrame:
        for _ in range(n_steps):
            self._simulate_step()

        return self.get_history

    def _output_constructor(self):
        out = pd.Series(self._out, name="output")
        out.index = out.index + self.n_lags  # Adjust for lagged inputs
        return out

    @property
    def get_states(self):
        return {
            k: v for k, v in enumerate(self.state_collection)
        }

    @property
    def get_history(self):
        return pd.concat([self.param_space_df, self._output_constructor()], axis=1)[5:].reset_index(drop=True)
