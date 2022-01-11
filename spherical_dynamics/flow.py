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
        val = 1
        w = np.prod(np.array(
            [self.lead_coefficient]
            + [np.Inf if z == p and order < 0
               else np.power(z - p, order)
               for p, order in self.zeros_with_orders]
        ))
        return cast(complex, w)

    def get_perturbed_zeros_with_orders(
        self,
        perturb_step_size: complex
    ) -> npt.NDArray[Tuple[complex, int]]:
        new_zeros = np.array(self.zeros_with_orders)
        for i, zero1, order1 in enumerate(self.zeros_with_orders):
            if np.absolute(order1) > 1:
                continue

            perturbation = perturb_step_size * self.lead_coefficient

            try:
                for j, zero2, order2 in enumerate(self.zeros_with_orders):
                    if i == j:
                        continue

                    perturbation *= np.power(zero1 - zero2, order2)
            except FloatingPointError as e:
                print(
                    f"Floating point error while perturbing zero at {zero1}"
                    + f"with order {order1}: leaving this zero unchanged"
                )
                continue

            new_zeros[i] = zero1 + perturbation, order1

        return new_zeros

    def perturb(self, perturb_step_size: complex) -> None:
        self.zeros_with_orders = self.get_perturbed_zeros_with_orders(perturb_step_size)