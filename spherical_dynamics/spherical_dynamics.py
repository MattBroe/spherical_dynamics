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
    resolution = 150
    num_curves = 60
    curve_graphs = [
        c.CurveGraph(
            np.array([
                (point, point.get_vector())
                for point in c.get_latitude_circle(resolution, n * np.pi / num_curves)
            ])
        )
        for n in np.arange(0, num_curves + 1)
    ]

    def generate_zero():
        arg = np.random.uniform(0, 2 * np.pi)
        modulus = np.random.uniform(0, 1)
        return np.exp(arg * 1j)

    first_zero = generate_zero()
    second_zero = generate_zero()
    zeros = np.array([first_zero, np.conj(first_zero), second_zero, np.conj(second_zero)])
    poly_flow = f.RationalFunctionFlow(1j, zeros)

    def complex_func(z: complex) -> complex:
        try:
            if np.isinf(z):
                return 0

            w = poly_flow.evaluate(z)
            if np.isinf(w) or w == 0:
                return 0

            w *= mu.bump(np.absolute(w/60))
            w = w / (1 + np.absolute(w))

            return w
        except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError):
            return 0

    plt.ion()
    figure = plt.figure()
    ax = plt.axes(projection='3d')

    for idx in np.arange(0, 500):
        plt.cla()
        for curve_graph in curve_graphs:
            for sphere_point, vec in curve_graph.get_points():
                try:
                    perturb_vector = sphere_point.evaluate_complex_function_as_tangent_vector(complex_func)
                except FloatingPointError:
                    continue
                for i, x in enumerate(vec):
                    try:
                        vec[i] = x + perturb_vector[i] * .1
                    except FloatingPointError:
                        continue

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
        poly_flow.perturb(.01)

    return

if __name__ == "__main__":
    main()
