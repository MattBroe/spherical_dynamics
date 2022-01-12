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
