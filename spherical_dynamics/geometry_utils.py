import numpy as np

zero_vector = np.zeros(3)
identity_matrix = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

def vector_equals(vec1, vec2):
    return (
        vec1[0] == vec2[0]
        and vec1[1] == vec2[1]
        and vec1[2] == vec2[2]
    )

def get_coordinates(vec):
    return vec[0], vec[1], vec[2]
    
def get_length(vec):
    x, y, z = get_coordinates(vec)
    return np.sqrt(x * x + y * y + z *z)

def get_direction(vec):
    r = get_length(vec)
    if r == 0:
        return zero_vector
    
    return np.array([coord / r for coord in vec])

#I hate spherical coordinates so we're doing everything based 
#on CCW rotations in xy and xz planes. The xy angle will lie in [0, pi)
#and the xz angle in [0, 2 * pi). 

def get_xy_angle(vec):
    unit_vec = get_direction(vec)
    x, y, _ = get_coordinates(unit_vec)
    if x == 0 and y == 0:
        return 0

    return np.arccos(x)

def get_xz_angle(vec):
    unit_vec = get_direction(vec)
    x, _, z = get_coordinates(unit_vec)
    if x == 0 and z == 0:
        return 0

    unsigned_angle = np.arccos(x)
    if z >= 0:
        return unsigned_angle

    return unsigned_angle + np.pi

def get_xy_rotation_matrix(angle):
    return np.array(
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    )

def get_xz_rotation_matrix(angle):
    return np.array(
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    )
    
def rotate_xy(matrix, angle):
    rotation = get_xy_rotation_matrix(angle)
    return np.matmult(rotation, matrix)

def rotate_xz(matrix, angle):
    rotation = get_xz_rotation_matrix(angle)
    return np.matmult(rotation, matrix)

def stereographic_project(unit_vec):
    x, y, z = get_coordinates(unit_vec)
    if z == 1:
        return np.array([np.Inf, np.Inf, 0])

    return np.array([x / (1 - z), y / (1 - z), 0])

#Often instead of working directly with
#a vector p lying on the unit sphere (away from the north and south poles),
#it will be more convenient to work with the rotation matrix which takes [1, 0, 0]
#to p.

def get_rotation_matrix(unit_vec):
    _, __, z = get_coordinates(unit_vec)
    if z == 1 or z == -1:
        raise ValueError("z was 1 or -1-- rotation matrix undefined")

    xy_angle = get_xy_angle(unit_vec)
    xz_angle = get_xz_angle(unit_vec)

    return rotate_xy(get_xz_rotation_matrix(xz_angle), xy_angle)

def get_unit_vector(rotation_matrix):
    return np.matmult(rotation_matrix, [1, 0, 0])

def is_north_pole(unit_vec):
    return unit_vec[2] == 1

def is_south_pole(unit_vec):
    return unit_vec[2] == -1

def is_north_or_south_pole(unit_vec):
    return is_north_pole(unit_vec) or is_south_pole(unit_vec)



