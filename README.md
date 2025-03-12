# MacroSim

`MacroSim` focuses on the derivation and iterative simulation of symbolic equations derived
through PySR's symbolic regression. Using well-known and simple economic models, MacroSim
brings the ability to experiment around the endogenization of model parameters.

## Example use cases

Using the well established Cobb-Douglas function as an example, we can traditionally treat
the CD equality as the function: $F(K, L, A, \alpha) = AK^\alpha \cdot L^{1-\alpha}$ where all components
of the function are parametrized instead of derived. For this specific example, we already have defined methodologies
of approximating (or calculating exactly) the parameters. Said methods were established through rigorous research over
time and widely accepted. However, such information may not be accessible for many other models of economic structures.

Taking the DAD-DAS model; we'll encounter a parameter of expected inflation which is often defined as: $ E[\pi_{t+1}] = \Theta(\pi_t) $. This definition may feel vague and undescriptive, the only information we are given is that there is some estiamtor $\Theta$ which (we assume) can perfectly estimate the expected inflation. This $\Theta$ is likely to vary depending on current economic state, time period, shocks, development levels, etc. and any sort of reliance to the DAD-DAS model requires some $\Theta$ to be derived from the ground up for each specific situation.

Symbolic regression coupled with a simulation engine can greatly reduce the human effort necessary to derive $\Theta$ in this example. By defining key variables, (for example: $\Delta_{CPI}$, $\Delta_{\frac{B}{Y}}$, $r$, $u$, etc.) collecting historical data, and fitting a symbolic regressor after cleaning the data. This approach is designed to maximize the control over how a symbolic representation (aka. function) should be derived. Size, complexity, allowed expression, elementwise loss, and many more parameters can be adjusted to derive best equation to model the target variable withind the given constraints. Of course, this is most often used to achieve interpretability; with the additional benefit of creating an opportunity to mathematically explore how predictions for a given variable are generated.

## Symbolic Regression Example

The symbolic regression backend of `MacroSim` relies on the `PySR` library which provides a symbolic regressor written in julia (a compiled language) and a python interface. The `macrosim.EqSearch` is a class that takes the `pysr.PySRRegressor` as its base and extends it by including model distillation and LOF outlier detection features. (Reasons behind opting for distillation and LOF based outlier removal are discussed further below)

We will demonstrate `EqSearch` by creating a fairly complex, yet mathematically accurate representation of the variable $L$ of the Cobb-Douglas production function. Data preparation steps are not included in this document, however all of the features, macroeconomic variables that are well-tracked my central banks and national statistic departments.

### Feature Set

* $\text{Total Population} := N$
* $\text{Post-Labor Population (Age>65)} := n_{>65}$
* $\text{Pre-Labor Population} := n_{<15}$
* $\text{Labor Force} := n_L$
* $\text{Labor Force Participation Rate} := \gamma$

### Target Variable

* $\text{Hours of Labor} := H_L$

A common common real-world definition of $L$ as a function is $L(H_L, w_h) = H_L \cdot w_h$ where $w_h$ represents the average (or median) hourly wage. You'll most likely notice that hours of total labor is a metric that is often recorded, tehrefore you might question reasoning behing endogenizing this variable. However, in a scenario where we're planning to extrapolate over a period of 20-50 year, an extremely accurate model of income generated through population is necessary. Raw demographic metrics can be modelled much more accurately through conventional practices and ML. Therefore, by having demographic metrics create the exogenous framework, we're essentially attempting to reorganize the CD parameters into a more simulation-friendly format.

### A Peak at the Results

To demostrate the outcome of the above describled experiment, we've defined a ran a regression process, converting all features to monthly frequencies, assuming unform distribution of quartlerly and annual variables over months. (This was done to artifically increase the dataset size without including pre-milenium data) To account for the high likelihood of overfitting due to perfectly unfiorm data distribution, a random normal noise factor, $\epsilon \sim N(0, \, 0.003 \cdot X_n)$ was added to each observation of the features that were subject to said frequency normalization. The outcome was a rather complex (the regression was run without complexity limitations), yet accurate expression:

$$
H_L(N, \ n_{>65},\ n_{<15},\ n_L,\ \gamma) = \gamma \cdot(sin(\gamma +0.27)cos(\gamma^{0.85})-2.40)\cdot(cos(0.01\sqrt{N})-2.73sin(\gamma)^2+26.27)+7.63e-5*N)
$$

Looking at the outcome, a valuable observation is that $n_{<15}$ and $n_{>65}$ were not used in the final expression. It is important to note that, `PySR` is designed to consider the simplicity of expressions and in case of equivalent accuracies, will select the equation with less parameters. Here we can reason about how $\gamma$ and $N$ were enough to derive a highly accurate output. Due to the seasonality and relative stability of the average hours worked per worker, knowing the population of labor force and their rate of particiaption accounts for all but one considerations; being seasonality. You'll notice an abundance of trigonometric functions, due to their cyclical behavior, these functions are the perfect candidate for modelling seasonality.

As you can see, we were able to extensively analyse and reason about the model output, which is simply not possible to this extent. (in non-linear cases) Moreover, we did not sacrifie a great deal of accuracy, as seen in the plot below, the model captured a good balance of sensitivity and generalization. (note that outliers were removed in training)

<p align="center">
  <img src="assets/LAB_HOURS_symbolic_pred.png" alt="Symbolic Regression Predictions for Total Hours of Labor">
</p>

### Generating Symbolic Expressions

Excluding data preprocessing, symbolic expressions can be generated through two method calls to an `EqSearch` instance. On the backend, `EqSearch` will remove local ouliers with a default contamination rate of $2.5\%$ and $n_{neighbors}=\lfloor n_{df}^{0.5}\rfloor$. Afterwards, a `sklearn.RandomForestRegressor` will be trained on the data and create predictions for the entire dataset. This step makes use of the robustness (aka. insensitivity to outliers) of the RandomForest algorithm to further distil the original labels. Through completing these steps, we aim to reach at a dataset where the features correspond to generalized labels instead of exact outcomes which generally increases the success rate of symbollic regression.



Knowing that the outcomes of symbolic regression (from PySR) are continuous and cannot be piecewise defined, you can imagine how attempting to fit to an ungeneralized set might turn out; therefore the safer approach of model distillation was picked as a design choice. This is the only additional functionality of `EqSearch` that builds on top of the regression model, therefore users who wish to opt out of distillation can directly utilise PySR and use the output in their simulations through `MacroSim`.


Regression outputs are generated with the code:

```python
from macrosim import EqSearch
import pandas as pd
from sympy import sin, cos

df = pd.read_csv(...)

#Prepare Data
...

X=df.drop('target', axis=1)
y=df['target'].to_frame()

eqsr = EqSearch(X=x, y=y)

eqsr.distil_split(grid_search=False) # To enable gridsearch for RandomForest, pass grid_search=True and param_grid={...}

eqsr.search(custom_loss='L2DistLoss()', # You can refer to PySR docs for predefined loss functions or define a custom 
                                        # function as a string using julia syntax

            extra_unary_ops={ # There are default lists of binary and unary operations, you cannot add custom binary operations, 
                              # however you can add unary operations using the format below.
                'cos2': {
                    'julia': 'cos2(x)=cos(x)^2',
                    'sympy': lambda x: cos(x) ** 2
                },
                'sin2': {
                    'julia': 'sin2(x)=sin(x)^2',
                    'sympy': lambda x: sin(x) ** 2
                }
            })

print(eqsr.eq) # 'eq' will contain the most accurate equation once EqSearch.search is called. Call EqSearch.sr._equations to get a 
               # DataFrame representing the whole search space.


```
