
from MacroSim.EqSearch import EqSearch

import pandas as pd
import numpy as np

from sympy import sin, cos

import matplotlib.pyplot as plt

if __name__ == "__main__":
    total_pop = pd.read_csv('./data/total_pop.csv')
    total_pop.set_index('observation_date', inplace=True)
    total_pop.index = pd.to_datetime(total_pop.index)
    total_pop.columns = ['TOT_POP']

    a65 = pd.read_csv('./data/A65_percent.csv')
    a65.set_index('observation_date', inplace=True)
    a65.index = pd.to_datetime(a65.index)
    a65.columns = ['A65']

    u14 = pd.read_csv('./data/U14_percent.csv')
    u14.set_index('observation_date', inplace=True)
    u14.index = pd.to_datetime(u14.index)
    u14.columns = ['U14']

    lab_hours = pd.read_csv('./data/lab_hours.csv')
    lab_hours.set_index('observation_date', inplace=True)
    lab_hours.index = pd.to_datetime(lab_hours.index)
    lab_hours.columns = ['LAB_HOURS']

    lab_force = pd.read_csv('./data/lab_force.csv')
    lab_force.set_index('observation_date', inplace=True)
    lab_force.index = pd.to_datetime(lab_force.index)
    lab_force.columns = ['LAB_FORCE']

    part_rate = pd.read_csv('./data/partrate.csv')
    part_rate.set_index('observation_date', inplace=True)
    part_rate.index = pd.to_datetime(part_rate.index)
    part_rate.columns = ['PART_RATE']

    df = pd.concat([total_pop, lab_hours, lab_force, part_rate], axis=1)
    df['TOT_POP'] = df['TOT_POP'].ffill().apply(lambda x: x + np.random.normal(0, 0.003*x)).round(1)
    df['LAB_HOURS'] = (df['LAB_HOURS'].ffill() / 12).apply(lambda x: x + np.random.normal(0, 0.003*x)).round(1)

    X = df.drop("LAB_HOURS", axis=1)
    y = df["LAB_HOURS"].to_frame()

    eqsr = EqSearch(X=X, y=y)

    eqsr.distil_split(grid_search=False)
    eqsr.search(custom_loss='L2DistLoss()',
                extra_unary_ops={
                    'cos2': {
                        'julia': 'cos2(x)=cos(x)^2',
                        'sympy': lambda x: cos(x)**2
                    },
                    'sin2': {
                        'julia': 'sin2(x)=sin(x)^2',
                        'sympy': lambda x: sin(x)**2
                    }
                })

    print(eqsr.eq)

    act = y.iloc[100:]
    pred = pd.DataFrame(eqsr.eq(X.iloc[100:]), index=y.iloc[100:].index)
    pred = pred

    plt.plot(act, label='Act', linestyle='--')
    plt.plot(pred, label='Pred', linestyle='-')

    plt.xticks(rotation=90)
    plt.legend()
    plt.show()
    plt.savefig('./test/L2 - eps_coef=0.003.png')
