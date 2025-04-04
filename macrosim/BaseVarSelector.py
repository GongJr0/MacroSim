import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import grangercausalitytests
from itertools import combinations
from statsmodels.tsa.api import VAR
from scipy.stats import f
from statsmodels.tools.sm_exceptions import MissingDataError
from statsmodels.tsa.ar_model import AutoReg
from typing import Any


class BaseVarSelector:
    def __init__(self, df):
        self.df = df
        self.score_dict: dict[str, Any] = {k: {} for k in self.df.columns}

    def granger_matrix(self):
        var_names = self.df.columns.tolist()
        n = len(var_names)
        G = pd.DataFrame(np.zeros((n, n)), index=var_names, columns=var_names)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                test_res = grangercausalitytests(self.df[[var_names[i], var_names[j]]],
                                                 maxlag=np.floor(np.sqrt(len(self.df))))

                p_val = [test_res[lag][0]['ssr_chi2test'][1] for lag in test_res]
                G.iloc[i, j] = min(p_val)

        return G

    def granger_score(self, G_matrix):
        outgoing = G_matrix.sum(axis=1)
        incoming = G_matrix.sum(axis=0)

        scores: pd.Series = (outgoing - incoming).round(4)
        ranks = scores.rank(ascending=True).astype(int)
        for col in G_matrix.columns:
            self.score_dict[col]['Granger'] = ranks[col]

    def multivar_granger_matrix(self):
        var_names = self.df.columns.tolist()
        scores = {var: [] for var in var_names}
        maxlag = int(np.floor(np.sqrt(len(self.df))))

        for predictors in combinations(var_names, 2):
            targets = [v for v in var_names if v not in predictors]

            for target in targets:
                full_vars = list(predictors) + [target]
                df_full = self.df[full_vars].dropna()

                if len(df_full) < (maxlag + 1) * len(full_vars):
                    continue

                try:
                    # Full model
                    full_model = VAR(df_full)
                    full_fit = full_model.fit(maxlags=maxlag)
                    resid_full = full_fit.resid[target]
                    rss_full = np.sum(resid_full**2)

                    # Restricted model
                    target_series = self.df[target].dropna()
                    if len(target_series) < (maxlag + 1):
                        continue

                    restricted_fit = AutoReg(target_series, lags=maxlag).fit()
                    resid_restricted = restricted_fit.resid
                    rss_restricted = np.sum(resid_restricted**2)

                    # Degrees of freedom
                    df_diff = full_fit.df_model - restricted_fit.df_model
                    n_obs = min(len(resid_full), len(resid_restricted))
                    if df_diff <= 0 or n_obs <= df_diff:
                        continue

                    # F-statistic
                    num = (rss_restricted - rss_full) / df_diff
                    denom = rss_full / (n_obs - full_fit.df_model)
                    f_stat = num / denom
                    p_value = 1 - f.cdf(f_stat, df_diff, n_obs - full_fit.df_model)

                    # Assign p-value to both predictors
                    scores[predictors[0]].append(p_value)
                    scores[predictors[1]].append(p_value)

                except (ValueError, MissingDataError, np.linalg.LinAlgError) as e:
                    print(f"Skipped {predictors} → {target} due to error: {e}")
                    continue

        scores = pd.Series([np.mean(scores[i]) for i in scores.keys()], index=scores.keys())
        ranks = scores.rank(ascending=True).astype(int)

        for col in ranks.index:
            self.score_dict[col]['Multivar_Granger'] = ranks[col]
