from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class SimState:
    # Production Function Modelling Params (Cobb-Douglas)

    # Capital Params
    A: float  # Capital Productivity (Percent)
    net_capital: float  # Currency
    alpha: float  # Share of capital (Percent)

    depreciation: float  # Percent
    capital_production: float  # Percent
    saving_rate: float  # Percent

    pop_tree: dict  # Keys: pre_labor, labor, post_labor (People)

    nat_growth: float  # Percent
    nat_decline: float  # Percent
    net_migration: float  # People
    lf_conversion_rate: float # Percent

    employment: float  # Percent
    labor_hours: float  # Hours per period per person
    hourly_wage: float  # Currency

    shock: bool  # True if economy in shock (i.e. x >= roc_threshold)

    prev_state: Optional[dict]  # dict(SimState) from previous iteration