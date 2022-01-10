import numpy as np


def bump(x):
    if np.absolute(x) >= 1:
        return 0

    return np.exp(-1 / (1 - x * x))
