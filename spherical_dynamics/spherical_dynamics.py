import numpy as np
import matplotlib.pyplot as plt
import jsons
import math_utils as mu
import spherical_point as sp
import geometry_utils as gu
import curves as c
import flow as f
import random_utils as r
from typing import Union
import time


class Configuration(object):
    def __init__(
        self,
        num_iterations: int,
        resolution: int,
        num_curves: int,
        flow_step_size: float,
        zero_perturb_size: float,
        zero_jump_probability: float,
        seed: int
    ):
        self.num_iterations = num_iterations
        self.resolution = resolution
        self.num_curves = num_curves
        self.flow_step_size = flow_step_size
        self.zero_perturb_size = zero_perturb_size
        self.zero_jump_probability = zero_jump_probability
        self.seed = seed

    def __str__(self) -> str:
        return jsons.dumps(self)


def main():
    np.seterr(all='raise')
    config = Configuration(500, 100, 14, .05, .1, .06, np.random.randint(0, 2 ** 31 - 1))
    print(f"Config: {config}")

    curve_graphs = [
        c.CurveGraph(
            np.array([
                (point, point.get_vector())
                for point in c.get_latitude_circle(config.resolution, n * np.pi / config.num_curves)
            ])
        )
        for n in np.arange(0, config.num_curves + 1)
    ]

    def generate_zero() -> complex:
        point = r.get_uniform_spherical_point()
        w = gu.stereographic_project_as_complex(point.get_vector())
        print(f"Random point on sphere: {point.get_vector()}. Stereographic projection: {w}")
        return w

    zeros_with_orders = np.array(
        ([(generate_zero(), 1) for _ in np.arange(0, 7)]
         + [(generate_zero(), -1) for _ in np.arange(0, 7)])
    )
    print(f"Zeros with orders: {zeros_with_orders}")

    perturb_gen = r.BimodalPerturbationGenerator(config.zero_perturb_size, config.zero_jump_probability)
    rational_flow = f.RationalFunctionFlow(1, zeros_with_orders)

    def complex_func(z: complex) -> complex:
        try:
            return rational_flow.evaluate(z)
        except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError):
            return np.inf

    plt.ion()
    figure = plt.figure()
    ax = plt.axes(projection='3d')

    for idx in np.arange(0, config.num_iterations):
        plt.cla()
        elev = ax.elev
        azim = ax.azim
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        zmin, zmax = ax.get_zlim()
        reverse = (idx // (config.num_iterations // 4)) % 2 == 0
        for curve_graph_idx, curve_graph in enumerate(curve_graphs):
            for sphere_point, vec in curve_graph.get_points():
                new_vec = np.empty(len(vec))
                try:
                    perturb_vector = sphere_point.evaluate_complex_function_as_point_on_sphere(
                        complex_func
                    ).get_vector()
                    if reverse:
                        perturb_vector *= -1

                    for i, x in enumerate(vec):
                        new_vec[i] = x + config.flow_step_size * perturb_vector[i]

                except FloatingPointError:
                    continue

                for i, y in enumerate(new_vec):
                    vec[i] = y

            xs, ys, zs = (
                [vec[0] for _, vec in curve_graph.get_points()],
                [vec[1] for _, vec in curve_graph.get_points()],
                [vec[2] for _, vec in curve_graph.get_points()]
            )
            color = "black" if curve_graph_idx % 2 == 0 else "#8b0000"
            ax.plot(xs, ys, zs, color)

        # for i, zero_with_order in enumerate(rational_flow.get_zeros_with_orders()):
        #     zero, order = zero_with_order
        #     zero_on_sphere = gu.inverse_stereographic_project(zero)
        #     zero_x, zero_y, zero_z = gu.get_coordinates(zero_on_sphere)
        #     ax.plot([zero_x], [zero_y], [zero_z])
        #     ax.text(zero_x, zero_y, zero_z, f"z{i}:{int(order)}", "x")

        ax.view_init(elev, azim)
        # ax.set_xlim(xmin, xmax)
        # ax.set_ylim(ymin, ymax)
        # ax.set_zlim(zmin, zmax)

        figure.canvas.draw()

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        figure.canvas.flush_events()

        rational_flow.perturb(
            perturb_gen.get_perturbation
        )

    return


if __name__ == "__main__":
    main()
