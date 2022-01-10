import numpy as np
import matplotlib.pyplot as plt
import math_utils as mu
import spherical_point as sp
import geometry_utils as gu
import curves as c
import flow as f
import time


def main():
    np.seterr(all='raise')
    resolution = 40
    num_curves = 15
    curve_graphs = [
        c.CurveGraph(
            np.array([
                (point, point.get_vector())
                for point in c.get_latitude_circle(resolution, n * np.pi / num_curves)
            ])
        )
        for n in np.arange(0, num_curves + 1)
    ]

    zeros = np.array([2, 2j, 5 + 5j])
    poly_flow = f.PolynomialFlow(1j, zeros, .01)

    def complex_func(z: complex) -> complex:
        try:
            if np.isinf(z):
                return 0

            w = poly_flow.evaluate(z)
            if np.isinf(w) or w == 0:
                return 0

            w *= mu.bump(np.absolute(w / 20))
            w = w / (1 + np.absolute(w))

            return w
        except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError):
            return 0

    plt.ion()
    figure = plt.figure()
    ax = plt.axes(projection='3d')

    for _ in np.arange(0, 500):
        plt.cla()
        for curve_graph in curve_graphs:
            for sphere_point, vec in curve_graph.get_points():
                perturb_vector = sphere_point.evaluate_complex_function_as_tangent_vector(complex_func)
                for i, x in enumerate(vec):
                    vec[i] = x + perturb_vector[i] * .1

            xs, ys, zs = (
                [vec[0] for _, vec in curve_graph.get_points()],
                [vec[1] for _, vec in curve_graph.get_points()],
                [vec[2] for _, vec in curve_graph.get_points()]
            )
            ax.plot(xs, ys, zs)
        figure.canvas.draw()

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        figure.canvas.flush_events()
        time.sleep(0.01)
        poly_flow.perturb()

    return

if __name__ == "__main__":
    main()
