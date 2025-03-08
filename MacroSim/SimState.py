from dataclasses import dataclass


@dataclass
class SimState:


    A: float  # Kapital Productivity
    c: float  # Propensity to consume

    pop_tree: dict  # Keys: pre_labor, labor, post_labor
    nat_growth: float  # Natural Population growth
