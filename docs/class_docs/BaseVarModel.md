# BaseVarModel
This class is responsible for modelling the variables selected as the base components of a feedback loop. `BaseVarModel`
is a class that extends `BaseVarSelector` and automates the process of base variable selection prior to modelling. The
class attempts to create an accurate symbolic regression (SR) model with lagged features of base variables and evaluates 
it against a random forest (RF) model.

The processing pipeline takes steps for aggressive outlier elimination and model distillation in the following structure:

1. Use `sklearn.neighbors.LocalOutlierFactor` to remove local outliers from the data
2. Check for seasonality in the filtered data through the use of autocorrelation analysis (ACF)
3. Fit a `sklearn.ensemble.RandomForestRegressor` to the filtered data, using lagged features.
4. Distill the data by replacing true y values by RF predictions. (further smooths out the noise)
5. Initiate a `PySRRegressor` instance and enable or disable cyclical trigonometric functions based on the seasonality
check at step 2 (trig enabled if data shows seasonality)
6. Fit the SR model to the distilled data
7. Compare $MSE_{SR}$ and $MSE_{RF}$, opt for using RF as the base model if SR introduces a significant increase to MSE

Following this pipeline, a generalized symbolic expression $f(x_{t}, \dots, x_{t-n}) = y_t$ is derived. (provided the 
equation produces reasonable MSE) Model selection at the final step is performed such that:

$$
E_{MSE} = \frac{MSE_{SR}}{MSE_{RF}} -1
$$

$$
\text{and}
$$

$$
Model = \begin{cases}
SR & \text{if  } E_{MSE} < 0.05 \\
RF & \text{if  } E_{MSE} \geq 0.05
\end{cases}
$$

## Example Usage
```python
from macrosim import BaseVarModel
import pandas as pd

df = pd.DataFrame(...)
bvm = BaseVarModel(df=df)

base: pd.DataFrame = bvm.get_base_candidates()

models = {}
for base_var in base.columns:
    bvm.symbolic_model(base[base_var],  # Run symbolic search with kwargs to set model params
                       maxsize=24,
                       niterations=200,
                       constraints= {
                           'atan': 1,  # Complexity of x for arctan(x)
                           '^': (-1, 3)  # Complexity of (x, y) for x^y. -1 = No constraint.
                       })  
    
    print(bvm.sr.get_best())  # Check descriptives of the best SR expression regardless of selected model 
    
    models[base_var] = bvm.model_select()  # Returns RF or SR based on MSE criteria
```

## Methods

### `BaseVarModel.get_best_candidates`
Run Granger Causality tests implemented in `BaseVarSelector` to determine the most causal variables.

__Params:__

- None

__Returns:__

- `pd.DataFrame`: Dataframe of the 2 most causal variables. (Will be parametrized to top $n$ variables in the future)

### `BaseVarModel.symbolic_model`
Fit an SR instance with the given base variable.

__Params:__

- `candidate: pd.Series`: Data of chosen base variable candidate

- `**kwargs`: Model parameters accessible through keyword arguments (refer to <a href="https://astroautomata.com/PySR/api/" target="_blank">PySR Docs</a> for further 
explanation of each kwarg)

  - `model_selection: Literal['best', 'score', 'accuracy'] = 'accuracy'`: Model selection criterion
  - `niterations: int = 300`: Iterations per cycle
  - `maxsize: int = 32`: Maximum size of the symbolic expression
  - `constraints: dict[str, int | tuple] = {}`: Extra constraints to the complexity of binary and unary operators
  - `elementwise_loss: str = 'L2DistLoss()'`: Loss function defined in julia syntax or one of the predefined loss functions
  available <a href="https://astroautomata.com/PySR/api/#the-objective" target="_blank">here</a>.
  - `progress: bool = False`: Enable progress bar (Does not work for ipython environments)
  - `temp_equation_file: bool = True`: Record search results to a csv if False.
  - `deterministic: bool = True`: Non-deterministic search if False, (parallelism='serial' is 
  required for deterministic behavior)
  - `parallelism: Literal['serial', 'multithreading', 'multiprocessing'] = 'serial`: Method of parallelization
  - `random_state: int = 0`: Randomness seed
  - `grid_search: False`: A grid search is performed at RF fit if True (gs_params should be passed for the grid 
  search to run)
  - `gs_params: dict[str, List[Any] = {}`: Parameter grid for the RF grid search