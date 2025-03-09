import pandas as pd

from MacroSim.SimState import SimState
from dataclasses import asdict

import numpy as np

class SimEngine:

    def __init__(self, init_state: SimState):

        self.state = init_state
        self.prev_state = ...

        self.y = pd.DataFrame

    def sim_loop(self, steps: int, warm_start: bool = False) -> None:
        Y = []
        for _ in range(steps):
            s = self.state

            A = s.A
            K = s.net_capital
            L = s.pop_tree['labor'] * s.employment * s.labor_hours
            a = s.alpha

            # Current State Cobb-Douglas
            y = A*(K**a)*(L**(1-a))

            try:
                scale = Y[-1]*0.02
            except:
                scale = y*0.02
            y += np.random.normal(loc=0, scale=scale)  # epsilon adjustment

            Y.append(y)

            self.prev_state = asdict(s)

            # Update state

            # Labor
            s.pop_tree['labor'] += (s.pop_tree['pre_labor'] * s.lf_conversion_rate + s.net_migration)
            s.pop_tree['pre_labor'] *= (1+s.nat_growth)
            s.pop_tree['post_labor'] *= (1-s.nat_decline)

            # Capital
            s.net_capital *= ((1+s.capital_production) * (1-s.depreciation))

        Y = pd.DataFrame(Y)
        if warm_start:
            self.y = pd.concat([self.y, Y])
        else:
            self.y = Y