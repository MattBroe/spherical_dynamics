import numpy as np
import numpy.typing as npt
from typing import Tuple, cast


class RationalFunctionFlow(object):
    def __init__(
        self,
        lead_coefficient: complex,
        zeros_with_orders: npt.NDArray[Tuple[complex, int]] # negative order for pole
    ):
        self.lead_coefficient = lead_coefficient
        self.zeros_with_orders = zeros_with_orders

    def evaluate(self, z: complex) -> complex:
        w = self.lead_coefficient
        if np.isinf(z):
            return w

        w *= np.prod(np.array(
           [np.Inf if z == p and order < 0
            else np.power(z - p, order)
            for p, order in self.zeros_with_orders]
        ))
        return cast(complex, w)

    def get_perturbed_zeros_with_orders(
        self,
        zero_perturb_step_size: complex,
        pole_perturb_step_size: complex
    ) -> npt.NDArray[Tuple[complex, int]]:
        new_zeros = np.copy(self.zeros_with_orders)
        for i, zero_with_order1 in enumerate(self.zeros_with_orders):
            zero1, order1 = zero_with_order1
            if np.absolute(order1) > 1:
                continue

            perturb_step_size = zero_perturb_step_size if order1 >= 0 else pole_perturb_step_size
            perturbation = perturb_step_size * self.lead_coefficient

            try:
                for j, zero_with_order2 in enumerate(self.zeros_with_orders):
                    zero2, order2 = zero_with_order2
                    if i == j:
                        continue
                    difference = zero1 - zero2
                    scale_factor = np.power(difference, order2)
                    perturbation *= scale_factor
            except FloatingPointError as e:
                print(
                    f"Floating point error while perturbing zero at {zero1}"
                    + f"with order {order1}: leaving this zero unchanged"
                )
                print(self.zeros_with_orders)
                raise e

            new_zeros[i] = zero1 + perturbation, order1

        return new_zeros

    def perturb(self, zero_perturb_step_size: complex, pole_perturb_step_size: complex) -> None:
        print(f"Start zeros: {self.zeros_with_orders}")
        perturbed_zeros = self.get_perturbed_zeros_with_orders(zero_perturb_step_size, pole_perturb_step_size)
        print(f"Perturbed zeros: {perturbed_zeros}")
        self.zeros_with_orders = perturbed_zeros
