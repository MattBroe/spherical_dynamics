import numpy as np
import SphericalPoint as sp
import geometry_utils as gu

def get_latitude_circle(resolution, xy_angle):
    latitude_circle = np.array([
        sp.SphericalPoint.create_from_angles(xy_angle, xz_angle) 
        for xz_angle in [n * np.pi / resolution for n in np.arange(0, 2 * resolution)]
    ])

    return latitude_circle




