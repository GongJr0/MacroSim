from numbers import Integral
from test_utils import is_integer_strict

from macrosim.Stationarity import StationarityResult


def test_stationarity_attrs(df):
    assert hasattr(df, 'stationarity'), 'API Extensions not loaded'
    assert isinstance(df.stationarity.results, StationarityResult), (
        f"Results mistype. Expected: {StationarityResult}, got {type(df.stationarity._results)}."
    )
    assert isinstance(df.stationarity.is_stationary, dict), (
        f"is_causal mistype. Expected: {dict}, got {type(df.stationarity.is_stationary)}."
    )
    assert is_integer_strict(df.causality.common_lag), (
        f"common_lag mistype. Expected: {Integral} base type, got {type(df.stationarity.common_lag)}."
    )