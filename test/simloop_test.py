from MacroSim.SimEngine import SimEngine
from MacroSim.SimState import SimState

import matplotlib.pyplot as plt
from PIL import Image
import io

def main():
    state = SimState(
        A=5.6,
        net_capital=69_000_000,
        alpha=0.4,
        depreciation=0.0005,
        capital_production=75_000/69_000_000,
        saving_rate=0.25,

        pop_tree={
            'pre_labor': 0.1759464 * 334_914_895,
            'labor': 170_359 * 1000,
            'post_labor': 0.1743182 * 334_914_895
        },

        nat_growth=0.01,
        nat_decline=0.005,

        net_migration=4_800_000,
        lf_conversion_rate=0.01-0.0002,

        employment=0.92,
        labor_hours=160,
        hourly_wage=35.93,

        shock=False,

        prev_state=None
    )

    engine = SimEngine(init_state=state, entropy_coef=0.33, birth_rate_sensitivity=0.001)
    engine.sim_loop(50, warm_start=False)

    buf = io.BytesIO()

    plt.plot(engine.y)
    plt.savefig(buf, format='png')

    buf.seek(0)
    img = Image.open(buf)
    img.show()


    print(engine.y.iloc[0])


if __name__ == '__main__':
    main()
