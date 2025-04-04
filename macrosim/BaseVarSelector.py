import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import entropy

class BaseVarSelector:
    def __init__(self, df):
        self.df = df

    def granger_matrix(self):
        var_names = self.df.columns.tolist()
        n = len(var_names)
        G = pd.DataFrame(np.zeros((n, n)), index=var_names, columns=var_names)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                test_res = grangercausalitytests(self.df[[var_names[i], var_names[j]]],
                                                 maxlag=np.floor(np.sqrt(len(self.df))),
                                                 verbose=False)

                p_val = [test_res[lag][0]['ssr_chi2test'][1] for lag in test_res]
                if min(p_val) < 0.05:
                    G.iloc[i, j] = 1

        return G

    @staticmethod
    def granger_score(G_matrix):
        outgoing = G_matrix.sum(axis=1)
        incoming = G_matrix.sum(axis=0)

        return (outgoing - incoming).sort_values(ascending=False)