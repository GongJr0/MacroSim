# -*- coding: utf-8 -*-

from dataclasses import asdict, make_dataclass

import sympy as sp
import pandas as pd
import numpy as np
import inspect

from io import StringIO
from copy import copy

from typing import Callable, Union, Any, Generator, Type


class SimEngine:
    """Class to iteratively simulate n periods of economic activity using a
    given symbolic equation and custom growth functions."""

    def __init__(self,
                 eq: sp.Expr,
                 init_params: dict[str, Union[Any, tuple[Any, Callable[[Any], Any]]]],
                 deterministic: bool = True,
                 entropy_coef: float = 0.025,
                 random_state: int = 42):
        """
        :param eq: Callable object derived from `macrosim.EqSearch`. Will be used to derive the primary generator object
        for iterative extrapolation.
        :param init_params: Parameters to define the starting point of the simulation and
        their respective (unary) growth functions as tuples. Dict format is {param_name: (param_val, growth_function)}.
        if ints are passed as values, no growth will be applied to parameters.
        :param deterministic: If True, `random_state` will be used to set the seed of all random generators.
        :param entropy_coef: Variance of the normal distribution, N(1, coef), to sample randomness. This is a multiplicative random term,
        meaning a coefficient of 0.1 would result in a \u00b1\\10% deviation from deterministic results on average.
        Set to 0 to disable random adjustments.
        :param random_state: Pseudo-random seed for periodic stochastic adjustment terms.
        """

        # Define function space and params
        self.eq = eq  # Lambda expression
        self._param_space: list[str] = list(init_params.keys())

        for param in init_params.keys():
            if type(init_params[param]) is int:
                init_params[param] = (init_params[param], lambda x: x)

        # Generate state recorder
        self._SimState = make_dataclass('SimState', [*self._param_space, 'output'])

        init_params['output'] = (0, None)
        self._prev = self._SimState(**{k: v[0] for k, v in init_params.items()})

        # Growth functions
        self.growth_functions: dict[str, Callable[[Any], Any]] = {k: v[1] for k,v in init_params.items()}


        # Entropy term generator
        if deterministic:
            np.random.seed(random_state)

        self.entropy = self.random_normal(entropy_coef)

        # CSV format simulation recorder
        self._hist = [",".join([*self._param_space, 'output'])]

    @staticmethod
    def random_normal(coef) -> Generator[float, None, None]:
        while True:
            yield np.random.normal(1, coef)

    def _simulate(self) -> Generator[Type[Any], None, None]:
        """
        A deterministic generator to backend to simulate n periods of
        economic activity based on symbolic function space.
        :return: `self.SimState`. `SimState` is a dynamically created dataclass
        that holds all params and generator output.
        """
        eq = self.eq
        prev = asdict(self._prev)

        prev_no_output = copy(prev)
        prev_no_output.pop('output', None)
        growth_functions = self.growth_functions
        while True:
            res = eq.subs(prev_no_output).evalf()
            param_eval = {
                k: growth_functions[k](v) * next(self.entropy)
                for k, v in prev_no_output.items()
            }
            param_eval.update({'output': res})

            self._hist.append(",".join(list(map(str, param_eval.values()))))
            self._prev = self._SimState(**param_eval)

            yield self._prev

    def get_history(self) -> pd.DataFrame:
        df = pd.read_csv(StringIO("\n".join(self._hist)), low_memory=False)
        df.index.name = 'step'
        return df
