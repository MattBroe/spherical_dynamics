import numpy as np
import matplotlib.pyplot as plt
import math_utils as mu
import spherical_point as sp
import geometry_utils as gu
import curves as c
import flow


def main():
    resolution = 50
    num_curves = 8
    curve_graphs = [
        c.CurveGraph(
            np.array([
                (point, point.get_vector())
                for point in c.get_latitude_circle(resolution, n * np.pi / num_curves)
            ])
        )
        for n in np.arange(0, num_curves + 1)
    ]

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for curve_graph in curve_graphs:
        xs, ys, zs = (
            [vec[0] for _, vec in curve_graph.points],
            [vec[1] for _, vec in curve_graph.points],
            [vec[2] for _, vec in curve_graph.points]
        )
        ax.plot(xs, ys, zs)

    plt.show()


if __name__ == "__main__":
    main()
