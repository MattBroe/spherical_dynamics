import numpy as np
import numpy.typing as npt
from typing import Tuple, cast, Callable
import geometry_utils as gu
import random_utils as r


class RationalFunctionFlow(object):
    def __init__(
        self,
        lead_coefficient: complex,
        zeros_with_orders: npt.NDArray[Tuple[complex, int]] # negative order for pole
    ):
        self.lead_coefficient = lead_coefficient
        self.zeros_with_orders = zeros_with_orders

    def get_zeros_with_orders(self):
        return np.copy(self.zeros_with_orders)

    def get_degree(self):
        return sum([zero_with_order[1] for zero_with_order in self.zeros_with_orders])

    def evaluate(self, z: complex) -> complex:
        w = self.lead_coefficient
        if np.isinf(z):
            degree = self.get_degree()
            if degree > 0:
                return z
            if degree < 0:
                return 0
            return w

        w *= np.prod(np.array(
           [np.Inf if z == p and order < 0
            else np.power(z - p, order)
            for p, order in self.zeros_with_orders]
        ))
        return cast(complex, w)

    def get_perturbed_lead_coefficient(
        self,
        get_perturbation: Callable[[], complex]
    ):
        inverse_stereo = gu.inverse_stereographic_project(self.lead_coefficient)
        perturbation = get_perturbation()
        new_inverse_stereo = gu.rotate_xy(gu.rotate_xz(inverse_stereo, perturbation.real), perturbation.imag)
        return gu.stereographic_project_as_complex(new_inverse_stereo)

    def get_perturbed_zeros_with_orders(
        self,
        get_perturbation: Callable[[], complex]
    ) -> npt.NDArray[Tuple[complex, int]]:
        new_zeros = np.copy(self.zeros_with_orders)
        for i, zero_with_order1 in enumerate(self.zeros_with_orders):
            zero1, order1 = zero_with_order1
            if np.absolute(order1) > 1:
                continue

            inverse_stereo = gu.inverse_stereographic_project(zero1)
            perturbation = get_perturbation()
            new_inverse_stereo = gu.rotate_xy(gu.rotate_xz(inverse_stereo, perturbation.real), perturbation.imag)
            print(f"Zero: {zero1} Perturbation: {perturbation}")
            new_zeros[i] = gu.stereographic_project_as_complex(new_inverse_stereo), order1

        return new_zeros

    def perturb(
        self,
        get_perturbation: Callable[[], complex],
        get_lead_coefficient_perturbation: Callable[[], complex],
    ) -> None:
        print(f"Start zeros: {self.zeros_with_orders}")
        perturbed_zeros = self.get_perturbed_zeros_with_orders(get_perturbation)
        print(f"Perturbed zeros: {perturbed_zeros}")
        self.zeros_with_orders = perturbed_zeros

        # TODO: find a good way to handle infinity. It needs to not just be a fixed point...
        # perturbed_lead_coefficient = self.get_perturbed_lead_coefficient(get_perturbation)
        # # if perturbed_lead_coefficient != 0:
        # #     perturbed_lead_coefficient /= np.absolute(perturbed_lead_coefficient)
        # #     perturbed_lead_coefficient *= np.absolute(self.lead_coefficient)
        #
        # self.lead_coefficient = perturbed_lead_coefficient
