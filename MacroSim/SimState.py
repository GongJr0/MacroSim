from dataclasses import dataclass


@dataclass
class SimState:

    A: float  # Capital Productivity
    c: float  # Propensity to consume

    pop_tree: dict  # Keys: pre_labor, labor, post_labor
    nat_growth: float  # Natural Population growth

    shock: bool  # True if economy in shock (i.e. x >= roc_threshold)

    prev_state: dict  # dict(SimState) from previous iteration
