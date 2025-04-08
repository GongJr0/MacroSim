# BaseVarSelector

`BaseVarSelector` handles variable selection for constructing the base of the feedback loop that models the variables' 
growth in the simulation process. It uses two implementations of __*Granger Causality Tests*__ to determine the variables
with the strongest causality over the whole feature set. The tests are of similar fashion with one being a __Bivariate GCT__
and the other being a __Multivariate GCT__. Both tests are structured tho check statistical significance of the statement
$H_0: x \underset{\text{Granger-cause}}{\not\to} z$ and $H_\alpha: x\underset{\text{Granger-cause}}{\to} z$. As any classical
hypothesis test, we check of the p-value of $H_0$. Lower p-values indicate greater statistical evidence against $H_0$, 
suggesting that **x Granger-causes z**. Importantly, GCT checks for linear causality so the significance we're computing 
is for the probability of a function existing such that:

$$
f(x_t, x_{t-1}, \dots, x_{t-n}) = z_t
$$

$$
\text{and}
$$

$$
f(x_t. x_{t-1}, \dots, x_{t-n}) = a_0x_t + a_1x_{t-1}+\dots+a_nx_{t-n}
$$

The multivariate case follow the same pattern and test if $(x, y) \underset{Granger-cause}{\to}z$. Unfortunately, we're
only testing for the existence of a function that satisfies the conditions defined above, therefore a multivariate test
will not provide context regarding which variable is more significant in creating causality. `BaseVarSelector` ranks 
variables (or pairs of variables) based on their p-values instead of using a threshold of significance to assert causality.
This ensures that the best performing variables will always be selected as opposed tests returning no eligible variables
for a given feature set.

## Example Usage
```python
from macrosim import BaseVarSelector
import pandas as pd

df = pd.DataFrame(...)
bvs = BaseVarSelector(df=df)

bvs.granger_matrix(score=True)  # Compute a matrix of p-values for all combinations of Bivariate GCTs; 
                                # record variable ranks on average p-value if score=True

bvs.multivar_granger_matrix()  # Compute p-values for all possible combinations of 2-predictor GCTs;
                               # record best variable ranks by average p-values of all pairs they've been a part of.
                               # (No matrix returned as there the raw data is often too large to visualise)

print(bvs.score_dict) # Print a dict listing the ranks of variables in terms of their performance (separate for both tests)

overall_score = {
    k: (2/3)*v['Granger'] + (1/3)*v['Multivar_Granger'] 
    for k, v in bvs.score_dict
}  # Weighted sum of scores from both tests for each variable
   # Lower is better
```

## Methods
### `BaseVarSelector.granger_matrix`
Computes a matrix of p-values for every possible __Bivariate GCT__ of the inputted data.

__Params:__

- `score: bool`: Record the ranks of variables into `BaseVarSelector.score_dict` if True.

__Returns:__

- `pd.DataFrame`: Matrix of p-values for all GCTs computed.

### `BaseVarSelector.multivar_granger_matrix`
Computes all __Multivariate GCTs__ with two predictors ($(x,y) \underset{\text{Granger-cause}}{\to}z$) and records the
variable ranks to `BaseVarSelector.score_dict`.

__Params:__

- None

__Returns:__

- None