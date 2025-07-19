from numbers import Integral


def is_integer_strict(x):
    return isinstance(x, Integral) and not isinstance(x, bool)