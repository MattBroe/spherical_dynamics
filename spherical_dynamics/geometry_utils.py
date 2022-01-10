import numpy as np
import numpy.typing as npt
from typing import Tuple

zero_vector = np.zeros(3)
identity_matrix = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
sphere_base_point = np.array([0, 0, -1])  # so the parallel transport singularity is at North pole


def vector_equals(vec1: npt.ArrayLike, vec2: npt.ArrayLike):
    return (
            len(vec1) == len(vec2)
            and all([x1 == x2 for x1, x2 in zip(vec1, vec2)])
    )


def get_coordinates(vec: npt.NDArray[float]) -> Tuple:
    return tuple(vec)


def get_length(vec: npt.NDArray[float]) -> float:
    return sum([x * x for x in vec])


def get_direction(vec: npt.NDArray[float]) -> npt.NDArray[float]:
    r = get_length(vec)
    if r == 0:
        return zero_vector

    return np.array([coord / r for coord in vec])


# I hate spherical coordinates so we're doing everything based
# on CCW rotations in xy and xz planes. The xy angle will lie in [0, pi)
# and the xz angle in [0, 2 * pi).

def get_xy_angle(vec: npt.NDArray[float]) -> float:
    unit_vec = get_direction(vec)
    x, y, *_ = get_coordinates(unit_vec)
    if x == 0 and y == 0:
        return 0

    return np.arccos(x)


def get_xz_angle(vec: npt.NDArray[float]) -> float:
    unit_vec = get_direction(vec)
    x, _, z, *_ = get_coordinates(unit_vec)
    if x == 0 and z == 0:
        return 0

    unsigned_angle = np.arccos(x)
    if z >= 0:
        return unsigned_angle

    return unsigned_angle + np.pi


def get_xy_rotation_matrix(angle: float) -> npt.NDArray[npt.NDArray[float]]:
    matrix = [
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ]
    np_matrix = np.array(matrix)
    return np_matrix
    # return np.array(
    #     [
    #         [np.cos(angle), np.sin(angle), 0],
    #         [-np.sin(angle), np.cos(angle), 0],
    #         [0, 0, 1]
    #     ]
    # )


def get_xz_rotation_matrix(angle: float) -> npt.NDArray[npt.NDArray[float]]:
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ]
    )


def rotate_xy(matrix: npt.ArrayLike, angle) -> npt.ArrayLike:
    rotation = get_xy_rotation_matrix(angle)
    return np.matmul(rotation, matrix)


def rotate_xz(matrix: npt.ArrayLike, angle: float) -> npt.ArrayLike:
    rotation = get_xz_rotation_matrix(angle)
    return np.matmul(rotation, matrix)


def stereographic_project(unit_vec: npt.NDArray[float]):
    x, y, z = get_coordinates(unit_vec)
    if z == 1:
        return np.array([np.Inf, np.Inf, 0])

    return np.array([x / (1 - z), y / (1 - z), 0])


__base_point_xy_angle = get_xy_angle(sphere_base_point)
__base_point_xz_angle = get_xz_angle(sphere_base_point)


def get_parallel_transport_matrix(unit_vec: npt.NDArray[float]) -> npt.NDArray[npt.NDArray[float]]:
    xy_angle = get_xy_angle(unit_vec)
    xz_angle = get_xz_angle(unit_vec)
    xy_angle_from_base_point = xy_angle + __base_point_xy_angle
    xz_angle_from_base_point = xz_angle + __base_point_xz_angle

    return rotate_xy(get_xz_rotation_matrix(xz_angle_from_base_point), xy_angle_from_base_point)


def get_unit_vector(parallel_transport_matrix: npt.NDArray[npt.NDArray[float]]) -> npt.NDArray[float]:
    return np.matmult(parallel_transport_matrix, sphere_base_point)


def is_north_pole(unit_vec: npt.NDArray[float]) -> bool:
    return unit_vec[2] == 1


def is_south_pole(unit_vec: npt.NDArray[float]) -> bool:
    return unit_vec[2] == -1


def is_north_or_south_pole(unit_vec: npt.NDArray[float]) -> bool:
    return is_north_pole(unit_vec) or is_south_pole(unit_vec)
