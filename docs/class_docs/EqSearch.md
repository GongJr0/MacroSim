# EqSearch
`EqSearch` is responsible with applying symbolic regression to derive target equations for simulation. 
It utilizes pysr's `PySRRegressor` model, which is written and pre-compiled in julia. As an important note,
pysr is currently in development and `UnicodeDecodeError`s are common during equation searches. 
(only tested in ipython environments)  Although raised, these byte decode errors do not interrupt the runtime and
better ipython support will likely be implemented at some point.

To generalize the output expressions, an aggressive outlier elimination process is applied in `EqSearch`. This increases
the chances of getting differentiable and interpretable outputs. n-neighbor based outlier detection is implemented with
`sklearn`'s `LocalOutlierFactor` model and after the LOF based outlier elimination, the remaining data is further distilled
through the use of a `RandomForestRegressor`. Following this procedure, `PySRRegressor` is utilized to conduct an iterative
search for the best fitting symbolic expression within the user-defined constraints.

## Example Usage
```python
from macrosim import EqSearch
from pandas import DataFrame, Series
from sympy import sin

x: DataFrame = ...
y: Series | DataFrame = ... 

eqsr = EqSearch(
    X=x,  # features
    y=y,  # label
    random_state=0,
    model_selection='best'
)

eqsr.distil_split(grid_search=False)  # Model distillation (required step)
eqsr.search(
    extra_unary_ops={
        'sin': {
            'julia': 'sin',
            'sympy': lambda x: sin(x)
        }
    },
    constraints={'sin': 2, '^': (-1, 1)} # Complexity limit of terms in binary and unary operations 
)

eq = eqsr.eq
```

# Methods

## `EqSearch.distil_split`
Applies outlier handling and model distillation.

__Params:__
- `test_size: float`: test size used for the data split of `RandomForestRegressor`.
- `grid_search: bool`: Conduct cross-validated grid search for `RandomForestRegressor` tuning if True.
- `gs_params: dict[str, list[Any]]`: Param grid to use if `grid_search=True`.

__Returns:__
- `None`

## `EqSearch.search`
Uses `PySRRegressor` to conduct a symbolic expression search. Calling `search` before `distil_split` will raise an
`AssertionError`. Records the resulting equation as a `sympy` expression to `self.eq`.

__Params:__
- `binary_ops: tuple[str]`: tuple of binary operators (as strings) allowed in the final expression. Defaults to use all 
binary operators available.

- `unary_ops: tuple[str]`: tuple of unary operators allowed in the final expression. Defaults to `('exp', 'log', 'sqrt')`.

- `extra_unary_ops: dict[str, dict[str, Any]]`: dict of unary ops that are not built-in to python, sympy, or julia. Every search
will include the extra unary: `{'inv': {julia: 'inv(x)=1/x', 'sympy': lambda x: 1/x}`. For each operator, a julia and sympy
applicable definition is necessary. julia implementations are passed as strings in julia syntax, to be parsed on the
`PySRRegressor` side.

- `custom_loss: str`: elementwise loss function to use, either written in julia syntax or a string literal for predefined loss
functions, available in [PySR documentation](https://ai.damtp.cam.ac.uk/symbolicregression/dev/losses/). Defaults to 'L2DistLoss()'. (sum of square difference, similar to MSE)

- `constraints: dict[str, Union[int, tuple[int, int]]`: constraints to expression complexity of binary and unary operations. 
Complexity, in this case, refers to the amount of operations required to reduce an input to its simplest form. For example, 
`1+1`can be reduced to it's simplest form in one addition, making it's overall complexity equal to 1 while `a*1+1` would 
require at least 2 operations, so it has a complexity of 2. (assuming a is a scalar) For unary operations, an integer is 
passed to define how complex of an input can be used. For example, a complexity of 1 would only allow a single variable 
or constant. With a complexity of 1, the `sin` operator would only be used as `sin(x)` while a  complexity of 2 would 
for example allow `sin(x+C)`. Binary operators function similarly, but their constraints are defined as a tuple of 
integers, for the inputs `x`, and `y` of the operation. For example, a constraint of `(1, 1)` on the `^` operator would 
only allow expressions of type `x^y` while a constraint of (2, 1) would allow `(x+1)^y`.

__Returns:__
- `None` (Results saved to `self.eq`)