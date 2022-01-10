import numpy as np
import numpy.typing as npt


class PolynomialFlow(object):
    def __init__(
        self,
        lead_coefficient: complex,
        zeros: npt.NDArray[complex],
        perturb_step_size: complex
    ):
        self.lead_coefficient = lead_coefficient
        self.zeros = zeros
        self.perturb_step_size = perturb_step_size

    def evaluate(self, z: complex):
        return np.prod(np.array([z - a for a in self.zeros] + [self.lead_coefficient]))

    def get_perturbed_zeros(self):
        new_zeros = np.array(self.zeros)
        for i, zero1 in enumerate(self.zeros):
            perturbation = np.complex(1 + 0j)
            for j, zero2 in enumerate(self.zeros):
                if i == j:
                    continue
                perturbation *= (zero1 - zero2)

            perturbation *= self.perturb_step_size
            new_zeros[i] = zero1 + perturbation

        return new_zeros

    def perturb(self):
        self.zeros = self.get_perturbed_zeros()
