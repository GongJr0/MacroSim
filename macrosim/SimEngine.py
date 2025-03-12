import pandas as pd

from macrosim.SimState import SimState
from dataclasses import asdict

import numpy as np

class SimEngine:

    def __init__(self, init_state: SimState, entropy_coef: float, birth_rate_sensitivity: float):

        self.state = init_state
        self.prev_state = ...

        self.y = pd.DataFrame()

        self.alpha_eps = entropy_coef

        self.base_gdp_pc = ...

        self.brs = birth_rate_sensitivity

        self.records = ...

    def scale_pop_growth(self, b0, y, p):
        return b0 * (1-(self.brs * ((y/p) / self.base_gdp_pc)))


    def scale_K(self, s, d, k0, y0):
        self.state.capital_production = s * (1-d)
        return s*y0 * (1-d)

    def sim_loop(self, steps: int, warm_start: bool = False) -> None:
        Y = []
        s = self.state
        eps_c = self.alpha_eps

        state_recs = {k: [] for k in list(asdict(s).keys())}

        for _ in range(steps):
            A = s.A + np.random.normal(0, 0.1*eps_c)
            K = s.net_capital + np.random.normal(0, s.net_capital*0.05*eps_c)
            L_base = (s.pop_tree['labor'] * s.employment * s.labor_hours * s.hourly_wage) / 1e6  # Convert to Million $
            L = L_base + np.random.normal(0, L_base*0.05*eps_c)
            a = s.alpha

            # Current State Cobb-Douglas
            y = A*(K**a)*(L**(1-a))

            Y.append(y)

            if len(Y) == 1:
                self.base_gdp_pc = Y[0]/sum(s.pop_tree.values())

            s.prev_state = asdict(s)

            # Update state
            for k in state_recs.keys():
                state_recs[k].append(asdict(s)[k])

            # Labor
            s.pop_tree['labor'] += (s.pop_tree['pre_labor'] * s.lf_conversion_rate + s.net_migration)
            s.pop_tree['pre_labor'] *= max((1+self.scale_pop_growth(s.nat_growth, y, sum(s.pop_tree.values()))), 0)
            s.pop_tree['post_labor'] *= (1-s.nat_decline)

            # Capital

            s.net_capital += self.scale_K(s.saving_rate, s.depreciation, K, y)

        Y = pd.DataFrame(Y)

        if warm_start:
            self.y = pd.concat([self.y, Y])
            self.records = pd.concat([self.records, pd.DataFrame(state_recs)])

        else:
            self.y = Y
            self.records = pd.DataFrame(state_recs)
