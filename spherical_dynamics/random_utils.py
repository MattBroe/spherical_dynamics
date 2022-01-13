import numpy as np
from spherical_point import SphericalPoint

def get_uniform_spherical_point() -> SphericalPoint:
    xy_angle = np.random.uniform(0, np.pi)
    xz_angle = np.random.uniform(0, 2 * np.pi)
    return SphericalPoint.create_from_angles(xy_angle, xz_angle)


def get_uniform_complex_on_unit_circle() -> complex:
    return np.exp(1j * np.random.uniform(0, 2 * np.pi))


def get_uniform_complex_in_disc(r: float) -> complex:
    return get_uniform_complex_on_unit_circle() * np.random.uniform(0, r)


class PerturbationGenerator(object):
    def __init__(self, step_size: float):
        self.step_size = step_size

    def get_perturbation(self):
        return self.step_size * get_uniform_complex_on_unit_circle() * np.random.lognormal(0, 1)


class BimodalPerturbationGenerator(object):
    def __init__(self, step_size: float, jump_probability: float):
        self.step_size = step_size
        self.jump_probability = jump_probability
        self.norm_params = [[step_size, step_size / 2], [10 * step_size, step_size / 2]]

    def get_perturbation(self):
        mixture_idx = np.random.choice(2, p=[1 - self.jump_probability, self.jump_probability])
        norm_params = self.norm_params[mixture_idx]
        if mixture_idx == 1:
            print("Jump!")
        magnitude = np.random.normal(norm_params[0], norm_params[1])
        return magnitude * get_uniform_complex_on_unit_circle()


