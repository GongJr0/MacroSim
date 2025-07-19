from numbers import Integral
from test_utils import is_integer_strict

from macrosim.Causality import CausalityResult


def test_causality_attrs(df):
    assert hasattr(df, 'causality'), 'API Extensions not loaded'
    assert isinstance(df.causality.results, CausalityResult), (
        f"Results mistype. Expected: {CausalityResult}, got {type(df.causality._results)}."
    )
    assert isinstance(df.causality.is_causal, dict), (
        f"is_causal mistype. Expected: {dict}, got {type(df.causality.is_causal)}."
    )
    assert is_integer_strict(df.causality.common_lag), (
        f"common_lag mistype. Expected: {Integral} base type, got {type(df.causality.common_lag)}."
    )
