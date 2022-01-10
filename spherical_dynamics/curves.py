import numpy as np
import numpy.typing as npt
import spherical_point as sp
from typing import Tuple


def get_latitude_circle(resolution: int, xy_angle: float) -> npt.NDArray[sp.SphericalPoint]:
    latitude_circle = np.array([
        sp.SphericalPoint.create_from_angles(xy_angle, xz_angle)
        for xz_angle in [n * np.pi / resolution for n in np.arange(0, 2 * resolution + 1)]
    ])

    return latitude_circle


class CurveGraph(object):
    def __init__(self, points: npt.NDArray[Tuple[sp.SphericalPoint, npt.NDArray[float]]]):
        self.__points = points

    def get_points(self) -> npt.NDArray[Tuple[sp.SphericalPoint, npt.NDArray[float]]]:
        return self.__points
