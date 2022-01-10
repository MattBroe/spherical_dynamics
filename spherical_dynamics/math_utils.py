import numpy as np


def bump(x: float) -> float:
    if np.absolute(x) >= 1:
        return 0

    try:
        return np.exp(-1 / (1 - x * x))
    except FloatingPointError:
        return 0
