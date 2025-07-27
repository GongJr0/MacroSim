# Args: [batch_filepath, batch_no, model_spec: Optional[Literal['True', 'False']] = 'True']
# Returns: None
import os
import pandas as pd
import numpy as np
from typing import Optional, cast, NewType, Callable, Any
from pysr import PySRRegressor, TemplateExpressionSpec, ExpressionSpec  # type: ignore
import sys
import pickle  # type: ignore
import re
import sympy as sp  # type: ignore

MODEL_LOCALS = {
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'exp': sp.exp,
    'log': sp.log,
    'sqrt': sp.sqrt,
    'pow': sp.Pow,  # Use SymPy's power function
}

ExprStr = NewType("ExprStr", str)

# Duplicate Functions from AutoReg.py to avoid import overhead
def get_expr(X: pd.DataFrame, target: str) -> TemplateExpressionSpec:
    var_set = [col for col in X.columns if col.startswith(target)]
    params = {"w": len(var_set)}

    sub_expr: ExprStr = cast(ExprStr,
                             " + ".join([f"w[{n + 1}] * {col}" for n, col in enumerate(var_set)])
                             )  # n+1 to match julia's 1-based indexing
    return TemplateExpressionSpec(
        expressions=["f"],
        variable_names=var_set,
        parameters=params,
        combine=f"f({sub_expr})"
    )


def model_config(X: pd.DataFrame, spec: Optional[TemplateExpressionSpec] = None) -> PySRRegressor:
    size = len(X.columns) #* 2
    spec = spec or ExpressionSpec()
    model = PySRRegressor(
        model_selection='best',

        niterations=50,
        maxsize=size,

        expression_spec=spec,

        binary_operators=['+', '-', '*', '/', 'pow'],
        unary_operators=['exp', 'log', 'sqrt', 'sin', 'cos', 'tan'],

        constraints={
            'sin': 2,
            'cos': 2,
            'tan': 2,

            'exp': 2,
            'log': 2,
            'sqrt': 3,
            'pow': (-1, 2)
        },
        nested_constraints={'sin': {'cos': 0, 'sin': 2},
                            'cos': {'sin': 0, 'cos': 2},
                            'tan': {'sin': 1, 'cos': 1},
                            'exp': {'log': 0},
                            'log': {'exp': 0},
                            },

        elementwise_loss='L2DistLoss()',

        temp_equation_file=True,
    )
    return model


def subs_expr(model: PySRRegressor) -> pd.Series:
    assert isinstance(model.expression_spec, TemplateExpressionSpec), "model.expression_spec must be a TemplateExpressionSpec"

    expr: TemplateExpressionSpec = model.expression_spec
    model_out: str = model.get_best()['equation']

    outer, coef = model_out.split(";")
    exec_str = coef.split("=")[1].strip()
    w: list[int] = eval(exec_str)
    assert isinstance(w, list), "w must be a list of coefficients"
    w_iter = iter(w)

    def replacer(match):
        return str(next(w_iter))

    func = expr.combine.replace("f(", "").replace(")", "")
    func_inner = re.sub(r"w\[\d+\]", replacer, func)

    replacements = [
        ("#1", func_inner),
        ("f =", ""),
        ("+-", "-"),
        ("-+", "-"),
    ]
    func_full = outer.replace("#1", func_inner).replace("f =", "").replace("+ -", "- ").replace("- +", "- ").strip()
    varnames = list(model.feature_names_in_)
    symbol_map = {name: sp.Symbol(name) for name in varnames}

    eq = sp.sympify(func_full, locals={**MODEL_LOCALS, **symbol_map})  # type: ignore
    model.equations_.loc[model.get_best().name, 'lambda_format'] = eq
    print(model.get_best())
    return model.get_best()


def fit(X: pd.DataFrame, y: pd.Series | pd.DataFrame, template_spec: bool = True) -> PySRRegressor:
    target = str(y.name) if isinstance(y, pd.Series) else y.columns[0]
    spec = get_expr(X=X, target=target) if template_spec else ExpressionSpec()

    spec = get_expr(X=X, target=target)

    model = model_config(X=X, spec=spec)
    model.fit(X=X, y=y)

    return model


if __name__ == '__main__':
    assert len(sys.argv) >= 3, "Expected at least 2 arguments: batch_filepath, batch_no"
    assert (batch_filepath := sys.argv[1]).endswith('.pkl'), "batch_filepath must be a .pkl file"
    assert (batch_no := int(sys.argv[2])) >= 0, "batch_no must be a non-negative integer"
    assert (template := sys.argv[3] if len(sys.argv) > 3 else 'True') in ('True', 'False'), "template must be 'True' or 'False'"

    template = eval(template)
    with open(batch_filepath, 'rb') as f:
        X, y = pickle.load(f)

    assert len(X) == len(y), "X and y must have same length"
    assert isinstance(y, pd.Series) or isinstance(y, pd.DataFrame), "y must be a pd.Series or pd.DataFrame"
    assert isinstance(X, pd.DataFrame), "X must be a pd.DataFrame"

    target = cast(str, y.name) if isinstance(y, pd.Series) else cast(str, y.columns[0])
    sr = fit(X=X, y=y, template_spec=template)
    eq_out = subs_expr(sr)

    if not os.path.exists('./batch/eq/'):
        os.makedirs('./batch/eq/', exist_ok=True)

    with open(f'./batch/eq/eq_{batch_no}.pkl', 'wb') as f:
        pickle.dump(eq_out, f)

    os.remove(batch_filepath)  # Clean up the batch file after processing
    exit(0)
