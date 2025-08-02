from test_utils import is_integer_strict
import pandas as pd


def test_data_split(autoreg):
    assert isinstance(autoreg.X, pd.DataFrame), f"X should be a DataFrame. Not {type(autoreg.X)}"


def test_batching(autoreg):
    batch = autoreg.shuffle_batch(2)
    assert isinstance(batch[0], enumerate), f"Batch should be an enumerate object. Not {type(batch[0])}"
    assert isinstance(batch[1][0], pd.DataFrame), (f"Test batch should contain a DataFrame at index 0. "
                                                   f"Not {type(batch[1][0])}")
    assert isinstance(batch[1][1], pd.Series), (f"Test batch should contain a Series at index 1. "
                                                f"Not {type(batch[1][1])}")


def test_lags(autoreg):
    assert is_integer_strict(autoreg.n_lags), f"n_lags should be an integer. Not {type(autoreg.n_lags)}"

    causals = autoreg.causal_lag_positions
    assert all(isinstance(key, str) for key in causals.keys()), "Causal lag positions keys should be strings."
    assert all(is_integer_strict(value) for value in causals.values()), "Causal lag positions values should be integers."


def test_freq(autoreg):
    assert is_integer_strict(autoreg.freq), f"freq should be an integer. Not {type(autoreg.freq)}"
