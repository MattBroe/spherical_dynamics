import geometry_utils as gu
import numpy as np


def parallel_transport(start_point, end_point, tangent_vec):
    start_matrix = start_point.get_rotation_matrix()
    if start_matrix is None:
        raise ValueError("start_point had undefined rotation matrix")

    end_matrix = start_point.get_rotation_matrix()
    if end_matrix is None:
        raise ValueError("end_point had undefined rotation matrix")

    return np.matmul(end_matrix, np.matmul(np.linalg.inv(start_matrix), tangent_vec))

class SphericalPoint(object):
    #Often instead of working directly with
    #a vector p lying on the unit sphere (away from the north and south poles),
    #it will be more convenient to work with the rotation matrix which takes [1, 0, 0]
    #to p.
    def get_vector(self):
        return self.__vector

    def get_rotation_matrix(self):
        return self.__rotation_matrix

    def is_north_pole(self):
        return self.get_vector()[2] == 1

    def is_south_pole(self):
        return self.get_vector()[2] == -1

    def is_north_or_south_pole(self):
        return self.is_north_pole() or self.is_south_pole()
    
    def __init__(self, unit_vec):
        self.__vector = unit_vec
        if self.is_north_or_south_pole():
            return

        self.__rotation_matrix = gu.get_rotation_matrix(unit_vec)

    basepoint = SphericalPoint(np.array([1, 0, 0]))

    def evaluate_complex_function_as_tangent_vector(self, complex_func):
        rotation_matrix = self.get_rotation_matrix()
        if rotation_matrix is None:
            return gu.zero_vector #TODO: replace with ValueError?

        proj = gu.stereographic_project(self.get_vector())
        complex_proj = np.complex(proj[0], proj[1])

        func_value = complex_func(complex_proj)
        func_value_tangent_vector_at_basepoint = np.array([func_value.real, 0, func_value.imag])
        func_value_tangent_vector = parallel_transport(basepoint, self, func_value_tangent_vector_at_basepoint)




