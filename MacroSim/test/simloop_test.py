from MacroSim.SimEngine import SimEngine
from MacroSim.SimState import SimState

import matplotlib.pyplot as plt


def main():
    state = SimState(
        A=0.9,
        net_capital=69_000_000,
        alpha=0.4,
        depreciation=0.0005,
        capital_production=75_000/69_000_000,

        pop_tree={
            'pre_labor': 0.1759464 * 334_914_895,
            'labor': 170_359 * 1000,
            'post_labor': 0.1743182 * 334_914_895
        },

        nat_growth=0.01,
        nat_decline=0.005,

        net_migration=4_800_000,
        lf_conversion_rate=0.01-0.0002,

        employment=0.96,
        labor_hours=160,

        shock=False,

        prev_state=None
    )

    engine = SimEngine(init_state=state)
    engine.sim_loop(50, warm_start=False)

    plt.plot(engine.y)
    plt.show()


if __name__ == '__main__':
    main()