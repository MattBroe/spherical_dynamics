import numpy as np
import numpy.typing as npt


class PolynomialFlow(object):
    def __init__(self, zeros: npt.ArrayLike[complex], perturb_step_size: complex):
        self.zeros = zeros

    def evaluate(self, z: complex):
        return np.prod(np.array([z - a for a in self.zeros]))