import numpy as np
import matplotlib.pyplot as plt
import math_utils as mu
import spherical_point as sp
import geometry_utils as gu
import curves as c
import flow as f
import random_utils as r
from typing import Union
import time


def main():
    np.seterr(all='raise')
    resolution = 150
    num_curves = 50
    flow_step_size = .1
    rational_function_zero_perturb_size = 50
    rational_function_pole_perturb_size = 5
    curve_graphs = [
        c.CurveGraph(
            np.array([
                (point, point.get_vector())
                for point in c.get_latitude_circle(resolution, n * np.pi / num_curves)
            ])
        )
        for n in np.arange(0, num_curves + 1)
    ]

    def generate_zero() -> complex:
        point = r.get_uniform_spherical_point()
        w = gu.stereographic_project_as_complex(point.get_vector())
        print(f"Random point on sphere: {point.get_vector()}. Stereographic projection: {w}")
        return w

    zeros_with_orders = np.array([
        (generate_zero(), 1),
        (generate_zero(), 1),
        (generate_zero(), -1),
        (generate_zero(), -1)
    ])
    print(f"Zeros with orders: {zeros_with_orders}")

    rational_flow = f.RationalFunctionFlow(1j, zeros_with_orders)

    def complex_func(z: complex) -> complex:
        try:
            return rational_flow.evaluate(z)
        except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError):
            return np.nan

    plt.ion()
    figure = plt.figure()
    ax = plt.axes(projection='3d')

    for _ in np.arange(0, 500):
        plt.cla()
        for curve_graph in curve_graphs:
            for sphere_point, vec in curve_graph.get_points():
                try:
                    perturb_vector = sphere_point.evaluate_complex_function_as_point_on_sphere(complex_func).get_vector()
                    for i, x in enumerate(vec):
                        vec[i] = x + flow_step_size * perturb_vector[i]

                except FloatingPointError:
                    continue

                # unit_vec = gu.get_direction(vec)
                # for i, _ in enumerate(vec):
                #     vec[i] = unit_vec[i]


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
        rational_flow.perturb(np.random.uniform(-1, 1), np.random.uniform(-1, 1))

    return


if __name__ == "__main__":
    main()
