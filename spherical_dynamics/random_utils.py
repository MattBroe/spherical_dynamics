import numpy as np
from spherical_point import SphericalPoint


def get_uniform_spherical_point() -> SphericalPoint:
    xy_angle = np.random.uniform(0, np.pi)
    xz_angle = np.random.uniform(0, 2 * np.pi)
    return SphericalPoint.create_from_angles(xy_angle, xz_angle)
