import numpy as np
import numpy.typing as npt
from typing import Callable, Union
import geometry_utils as gu


class SphericalPoint(object):
    pass


def get_basepoint() -> SphericalPoint:
    pass


class SphericalPoint(object):
    # Often instead of working directly with
    # a vector p lying on the unit sphere
    # it will be more convenient to work with the parallel_transport matrix which takes the basepoint
    # to p (along a geodesic). Obviously not uniquely defined at antipode...

    # Intended as private constructor...
    def __init__(self, unit_vec: npt.NDArray[float], parallel_transport_matrix: npt.ArrayLike):
        self.__vector = unit_vec
        self.__parallel_transport_matrix = parallel_transport_matrix

    def get_vector(self) -> npt.NDArray[float]:
        return np.copy(self.__vector)

    def get_parallel_transport_matrix(self) -> npt.ArrayLike:
        return np.copy(self.__parallel_transport_matrix)

    @staticmethod
    def create_from_unit_vector(unit_vec: npt.NDArray[float]) -> SphericalPoint:
        parallel_transport_matrix = gu.get_parallel_transport_matrix(unit_vec)
        return SphericalPoint(unit_vec, parallel_transport_matrix)

    @staticmethod
    def create_from_angles(xy_angle: float, xz_angle: float):
        parallel_transport_matrix = gu.rotate_xy(gu.get_xz_rotation_matrix(xz_angle), xy_angle)
        vector = np.matmul(parallel_transport_matrix, get_basepoint().get_vector())
        return SphericalPoint(vector, parallel_transport_matrix)

    def evaluate_complex_function_as_tangent_vector(
            self,
            complex_func: Callable[[complex], complex]
    ) -> npt.NDArray[float]:
        parallel_transport_matrix = self.get_parallel_transport_matrix()
        if parallel_transport_matrix is None:
            return gu.zero_vector  # TODO: replace with ValueError?

        proj = gu.stereographic_project_as_complex(self.get_vector())

        func_value = complex_func(proj)
        func_value_tangent_vector_at_basepoint = np.array([func_value.real, func_value.imag, 0])
        func_value_tangent_vector = parallel_transport(
            get_basepoint(),
            self,
            func_value_tangent_vector_at_basepoint
        )
        return func_value_tangent_vector

    def evaluate_complex_function_as_point_on_sphere(
        self,
        complex_func: Callable[[complex], complex]
    ) -> Union[SphericalPoint, None]:
        proj = gu.stereographic_project_as_complex(self.get_vector())
        func_value = complex_func(proj)
        if np.isnan(func_value) and not np.isinf(func_value):
            return self

        image_vector_on_sphere = gu.inverse_stereographic_project(func_value)
        return SphericalPoint.create_from_unit_vector(image_vector_on_sphere)


def parallel_transport(start_point: SphericalPoint, end_point: SphericalPoint, tangent_vec: npt.ArrayLike):
    start_matrix = start_point.get_parallel_transport_matrix()
    end_matrix = end_point.get_parallel_transport_matrix()

    return np.matmul(end_matrix, np.matmul(np.linalg.inv(start_matrix), tangent_vec))


__basepoint = SphericalPoint(gu.sphere_base_point, gu.identity_matrix)


def get_basepoint() -> SphericalPoint:
    return __basepoint
