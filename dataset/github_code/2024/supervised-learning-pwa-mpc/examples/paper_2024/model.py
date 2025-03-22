from typing import Literal

import numpy as np

from slpwampc.core.systems import PwaSystem
from slpwampc.misc.sampling import grid_sample_region, random_sample_region


class Model:
    nx = 2  # state dimension
    nu = 1  # control dimension
    num_ineqs = 1  # number of inequalities defining the pwa regions
    l = 2  # number of regions

    # pwa regions defined by Sx + Ru <= T
    S = [np.array([[1, 0]]), np.array([[-1, 0]])]
    R = [np.zeros((1, nu)), np.zeros((1, nu))]
    T = [np.array([[1]]), np.array([[-1]])]

    # x^+ = Ax + Bu + c
    A = [np.array([[1, 0.2], [0, 1]]), np.array([[0.5, 0.2], [0, 1]])]
    B = [np.array([[0.1], [1]]), np.array([[0.1], [1]])]
    c = [np.zeros((nx, 1)), np.array([[0.5], [0]])]

    # state constraints Dx <= E
    D = np.array([[-1, 1], [-3, -1], [0.2, 1], [-1, 0], [1, 0], [0, -1]])
    E = np.array([[15], [25], [9], [6], [8], [10]])

    # constrol constraints Fu <= G
    F = np.array([[1], [-1]])
    u_lim = 3
    G = u_lim * np.array([[1], [1]])

    system = {
        "S": S,
        "R": R,
        "T": T,
        "A": A,
        "B": B,
        "c": c,
        "D": D,
        "E": E,
        "F": F,
        "G": G,
    }

    X_f: tuple[np.ndarray, np.ndarray] = (
        np.array(
            [
                [0.943554152340661, 0.126216752413879],
                [-0.564458476593388, -0.737832475861210],
                [0.564458476593388, 0.737832475861210],
                [1, 0],
                [-1, 0],
            ]
        ),
        np.array([[1, 2, 2, 1, 6]]).T,
    )

    K_term = np.array([[-0.564458476593388, -0.737832475861210]])

    @staticmethod
    def get_system_dict():
        return Model.system

    @staticmethod
    def get_system():
        return PwaSystem(Model.system)

    @staticmethod
    def sample_state_space(
        np_random: np.random.Generator,
        sample_strategy: Literal["random", "grid", "focused"] = "random",
        num_points: int = 100,
        d: float = 0.1,
    ) -> list[np.ndarray]:
        """Sample points from the state space of the system.

        Parameters
        ----------
        np_random : np.random.Generator
            The random number generator.
        sample_strategy : Literal["random", "grid", "focused"], optional
            The strategy to sample points. If random, num_points points are sampled uniformly at random from the state space.
            If grid, points are sampled on a grid with spacing d.
            If focused, points are sampled in regions of the state space where the sboundaries are. By default "random".
        num_points : int, optional
            The number of points to sample for random strategy, by default 100.
        d : float, optional
            The spacing between grid points for grid strategy, by default 0.1.
        """
        if sample_strategy == "random":
            return random_sample_region(Model.D, Model.E, num_points, np_random)
        elif sample_strategy == "grid":
            regions_points = [
                grid_sample_region(np.vstack((S, Model.D)), np.vstack((T, Model.E)), d)
                for S, T in zip(Model.S, Model.T)
            ]
            return np.concatenate(
                regions_points,
                axis=0,
            )
        else:
            raise NotImplementedError()
