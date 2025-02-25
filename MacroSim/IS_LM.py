from SimState import SimState
from typing import Any

class IS_LM:
    def __init__(self, state: SimState, state_prev: dict[str, Any]):
        self.state = state
        self.prev = state_prev

        # Global Indicators
        self.gdp = state.gdp_nom
        self.rgdp = state.gdp_real

        # IS Curve components
        self.c = state.consumption
        self.t = state.avg_tax
        self.s = state.savings
        self.g = state.gov_exp
        self.nx = state.nx
        self.i = state.fed_interest

        # LM Curve Components
        self.m = state.money_supply
        self.rw = state.real_wage
        self.p = self.cpi_pred(
            self.m,
            self.i,
            self.rgdp,
            self.rw
        )

    def cpi_pred(self, m, i, rgdp, rwage) -> float:
        ...
    