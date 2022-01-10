import numpy as np
import numpy.typing as npt
import spherical_point as sp
from typing import Tuple


def get_latitude_circle(resolution: float, xy_angle: float) -> npt.ArrayLike[sp.SphericalPoint]:
    latitude_circle = np.array([
        sp.SphericalPoint.create_from_angles(xy_angle, xz_angle)
        for xz_angle in [n * np.pi / resolution for n in np.arange(0, 2 * resolution)]
    ])

    return latitude_circle


class CurveGraph(object):
    def __init__(self, points: npt.ArrayLike[Tuple[sp.SphericalPoint, npt.NDArray[3, float]]]):
        self.points = points
