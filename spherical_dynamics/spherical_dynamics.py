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
    resolution = 300
    num_curves = 12
    flow_step_size = .05
    zero_perturb_size = .6
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
        (generate_zero(), 1),
        (generate_zero(), -1),
        (generate_zero(), -1),
        (generate_zero(), -1),
    ])
    print(f"Zeros with orders: {zeros_with_orders}")

    rational_flow = f.RationalFunctionFlow(1, zeros_with_orders)

    def complex_func(z: complex) -> complex:
        try:
            return rational_flow.evaluate(z)
        except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError):
            return np.nan

    plt.ion()
    figure = plt.figure()
    ax = plt.axes(projection='3d')

    for idx in np.arange(0, 500):
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

        for i, zero_with_order in enumerate(rational_flow.get_zeros_with_orders()):
            zero, order = zero_with_order
            zero_on_sphere = gu.inverse_stereographic_project(zero)
            zero_x, zero_y, zero_z = gu.get_coordinates(zero_on_sphere)
            ax.plot([zero_x], [zero_y], [zero_z])
            ax.text(zero_x, zero_y, zero_z, f"z{i}:{order}", "x")


        figure.canvas.draw()

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        figure.canvas.flush_events()

        zero_perturbation = r.get_uniform_complex_in_disc(zero_perturb_size)
        rational_flow.perturb(
            zero_perturbation,
            zero_perturbation
        )

    return


if __name__ == "__main__":
    main()
