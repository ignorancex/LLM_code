#! /usr/bin/env python3

r"""
Multi-objective optimization benchmark problems.

Adapted some problems from https://github.com/ryojitanabe/reproblems
"""

from __future__ import annotations

from math import pow, sqrt

import torch
from botorch.test_functions.base import MultiObjectiveTestProblem
from torch import Tensor


class MarineDesign(MultiObjectiveTestProblem):
    r"""Conceptual marine design.

    Adapted problem `RE32` in https://github.com/ryojitanabe/reproblems
    """
    dim = 6
    num_objectives = 4
    _bounds = [
        (150.0, 274.32),
        (20.0, 32.31),
        (13.0, 25.0),
        (10.0, 11.71),
        (14.0, 18.0),
        (0.63, 0.75),
    ]
    _ref_point = [-250.0, 20000.0, 25000.0, 15.0]
    _num_original_constraints = 9

    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.
        """
        x_L = X[:, 0]
        x_B = X[:, 1]
        x_D = X[:, 2]
        x_T = X[:, 3]
        x_Vk = X[:, 4]
        x_CB = X[:, 5]

        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g = 9.8065
        Fn = V / torch.pow(g * x_L, 0.5)
        a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
        b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

        power = (torch.pow(displacement, 2.0 / 3.0) * torch.pow(x_Vk, 3.0)) / (
            a + (b * Fn)
        )
        outfit_weight = (
            1.0
            * torch.pow(x_L, 0.8)
            * torch.pow(x_B, 0.6)
            * torch.pow(x_D, 0.3)
            * torch.pow(x_CB, 0.1)
        )
        steel_weight = (
            0.034
            * torch.pow(x_L, 1.7)
            * torch.pow(x_B, 0.7)
            * torch.pow(x_D, 0.4)
            * torch.pow(x_CB, 0.5)
        )
        machinery_weight = 0.17 * torch.pow(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * (
            (2000.0 * torch.pow(steel_weight, 0.85))
            + (3500.0 * outfit_weight)
            + (2400.0 * torch.pow(power, 0.8))
        )
        capital_costs = 0.2 * ship_cost

        DWT = displacement - light_ship_weight

        running_costs = 40000.0 * torch.pow(DWT, 0.3)

        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_Vk
        handling_rate = 8000.0

        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * torch.pow(DWT, 0.8)

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * torch.pow(DWT, 0.5)

        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA

        f1 = annual_costs / annual_cargo
        f2 = light_ship_weight
        f3 = -annual_cargo

        # Reformulated objective functions
        g1 = (x_L / x_B) - 6.0
        g2 = -(x_L / x_D) + 15.0
        g3 = -(x_L / x_T) + 19.0
        g4 = 0.45 * torch.pow(DWT, 0.31) - x_T
        g5 = 0.7 * x_D + 0.7 - x_T
        g6 = 500000.0 - DWT
        g7 = DWT - 3000.0
        g8 = 0.32 - Fn

        g = torch.stack([g1, g2, g3, g4, g5, g6, g7, g8], dim=-1)
        g = torch.where(g < 0, -g, torch.zeros(g.size()))

        f4 = torch.sum(g, dim=-1)

        return torch.stack([f1, f2, f3, f4], dim=-1)


class RocketInjector(MultiObjectiveTestProblem):
    r"""Rocket injector design.

    Adapted problem `RE37` in https://github.com/ryojitanabe/reproblems
    """
    dim = 4
    num_objectives = 3
    _bounds = [
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ]
    _ref_point = [1.05, 1.3, 1.2]
    _num_original_constraints = 0

    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.
        """
        xAlpha = X[..., 0]
        xHA = X[..., 1]
        xOA = X[..., 2]
        xOPTT = X[..., 3]

        # f1 (TF_max)
        f1 = (
            0.692
            + (0.477 * xAlpha)
            - (0.687 * xHA)
            - (0.080 * xOA)
            - (0.0650 * xOPTT)
            - (0.167 * xAlpha * xAlpha)
            - (0.0129 * xHA * xAlpha)
            + (0.0796 * xHA * xHA)
            - (0.0634 * xOA * xAlpha)
            - (0.0257 * xOA * xHA)
            + (0.0877 * xOA * xOA)
            - (0.0521 * xOPTT * xAlpha)
            + (0.00156 * xOPTT * xHA)
            + (0.00198 * xOPTT * xOA)
            + (0.0184 * xOPTT * xOPTT)
        )
        # f2 (X_cc)
        f2 = (
            0.153
            - (0.322 * xAlpha)
            + (0.396 * xHA)
            + (0.424 * xOA)
            + (0.0226 * xOPTT)
            + (0.175 * xAlpha * xAlpha)
            + (0.0185 * xHA * xAlpha)
            - (0.0701 * xHA * xHA)
            - (0.251 * xOA * xAlpha)
            + (0.179 * xOA * xHA)
            + (0.0150 * xOA * xOA)
            + (0.0134 * xOPTT * xAlpha)
            + (0.0296 * xOPTT * xHA)
            + (0.0752 * xOPTT * xOA)
            + (0.0192 * xOPTT * xOPTT)
        )
        # f3 (TT_max)
        f3 = (
            0.370
            - (0.205 * xAlpha)
            + (0.0307 * xHA)
            + (0.108 * xOA)
            + (1.019 * xOPTT)
            - (0.135 * xAlpha * xAlpha)
            + (0.0141 * xHA * xAlpha)
            + (0.0998 * xHA * xHA)
            + (0.208 * xOA * xAlpha)
            - (0.0301 * xOA * xHA)
            - (0.226 * xOA * xOA)
            + (0.353 * xOPTT * xAlpha)
            - (0.0497 * xOPTT * xOA)
            - (0.423 * xOPTT * xOPTT)
            + (0.202 * xHA * xAlpha * xAlpha)
            - (0.281 * xOA * xAlpha * xAlpha)
            - (0.342 * xHA * xHA * xAlpha)
            - (0.245 * xHA * xHA * xOA)
            + (0.281 * xOA * xOA * xHA)
            - (0.184 * xOPTT * xOPTT * xAlpha)
            - (0.281 * xHA * xAlpha * xOA)
        )

        return torch.stack([f1, f2, f3], dim=-1)


class FourBarTrussDesign(MultiObjectiveTestProblem):
    r"""Four bar truss design.

    Adapted problem `RE21` in https://github.com/ryojitanabe/reproblems
    """
    dim = 4
    num_objectives = 2
    _bounds = [
        (1.0, 3.0),
        (sqrt(2.0), 3.0),
        (sqrt(2.0), 3.0),
        (1.0, 3.0),
    ]
    _ref_point = [3000.0, 0.05]
    _num_original_constraints = 0

    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.
        """
        x1 = X[..., 0]
        x2 = X[..., 1]
        x3 = X[..., 2]
        x4 = X[..., 3]

        F = 10.0
        E = 2.0 * 1e5
        L = 200.0

        f1 = L * ((2 * x1) + sqrt(2.0) * x2 + torch.sqrt(x3) + x4)
        f2 = ((F * L) / E) * (
            (2.0 / x1) + (2.0 * sqrt(2.0) / x2) - (2.0 * sqrt(2.0) / x3) + (2.0 / x4)
        )

        return torch.stack([f1, f2], dim=-1)


class TwoBarTrussDesign(MultiObjectiveTestProblem):
    r"""Two bar truss design.

    Adapted problem `RE31` in https://github.com/ryojitanabe/reproblems
    """
    dim = 3
    num_objectives = 3
    _bounds = [
        (0.00001, 100.0),
        (0.00001, 100.0),
        (1.0, 3.0),
    ]
    _ref_point = [815.0, 4000000.0, 8600000.0]
    _num_original_constraints = 3

    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.
        """

        x1 = X[..., 0]
        x2 = X[..., 1]
        x3 = X[..., 2]

        f1 = x1 * torch.sqrt(16.0 + (x3 * x3)) + x2 * torch.sqrt(1.0 + x3 * x3)
        f2 = (20.0 * torch.sqrt(16.0 + (x3 * x3))) / (x1 * x3)

        # Constraint functions
        g1 = 0.1 - f1
        g2 = 100000.0 - f2
        g3 = 100000 - ((80.0 * torch.sqrt(1.0 + x3 * x3)) / (x3 * x2))

        g = torch.stack([g1, g2, g3], dim=-1)
        g = torch.where(g < 0, -g, torch.zeros(g.size()))
        f3 = torch.sum(g, dim=-1)

        return torch.stack([f1, f2, f3], dim=-1)


class WeldedBeam(MultiObjectiveTestProblem):
    r"""Welded beam design.

    Adapted problem `RE32` in https://github.com/ryojitanabe/reproblems
    """

    dim = 4
    num_objectives = 3
    _bounds = [
        (0.125, 5.0),
        (0.1, 10.0),
        (0.1, 10.0),
        (0.125, 5.0),
    ]
    _ref_point = [330.0, 17000.0, 400000000.0]
    _num_original_constraints = 4

    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.
        """
        x1 = X[..., 0]
        x2 = X[..., 1]
        x3 = X[..., 2]
        x4 = X[..., 3]

        P = 6000
        L = 14
        E = 30 * 1e6
        G = 12 * 1e6
        tau_max = 13600
        sigma_max = 30000

        f1 = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        f2 = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)

        M = P * (L + (x2 / 2))
        R = torch.sqrt(((x2 * x2) / 4.0) + torch.pow((x1 + x3) / 2.0, 2))
        J = 2 * sqrt(2) * x1 * x2 * ((x2 * x2) / 12.0) + torch.pow((x1 + x3) / 2.0, 2)

        tau_1 = P / (sqrt(2) * x1 * x2)
        tau_2 = (M * R) / J
        tau = torch.sqrt(
            tau_1 * tau_1 + ((2 * tau_1 * tau_2 * x2) / (2 * R)) + (tau_2 * tau_2)
        )
        sigma = (6 * P * L) / (x4 * x3 * x3)
        PC = (
            4.013
            * E
            * torch.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0)
            / (L * L)
            * (1 - (x3 / (2 * L)) * sqrt(E / (4 * G)))
        )

        g1 = tau_max - tau
        g2 = sigma_max - sigma
        g3 = x4 - x1
        g4 = PC - P

        g = torch.stack([g1, g2, g3, g4], dim=-1)
        g = torch.where(g < 0, -g, torch.zeros(g.size()))
        f3 = torch.sum(g, dim=-1)

        return torch.stack([f1, f2, f3], dim=-1)


class CabDesign(MultiObjectiveTestProblem):
    r"""Car cab design.

    Adapted problem `RE91` in https://github.com/ryojitanabe/reproblems
    """
    dim = 7
    num_objectives = 9
    _bounds = [
        (0.5, 1.5),
        (0.45, 1.35),
        (0.5, 1.5),
        (0.5, 1.5),
        (0.875, 2.625),
        (0.4, 1.2),
        (0.4, 1.2),
    ]
    _ref_point = [42.0, 1.05, 0.95, 0.8, 1.5, 1.15, 1.15, 1.05, 1.05]
    _num_original_constraints = 0

    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.
        """
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]
        # Stochastic variables
        # Note we set the stochastic parameter to zero.
        Z = 0
        x8 = 0.006 * Z + 0.345
        x9 = 0.006 * Z + 0.192
        x10 = 10 * Z + 0.0
        x11 = 10 * Z + 0.0

        f1 = (
            1.98
            + 4.9 * x1
            + 6.67 * x2
            + 6.98 * x3
            + 4.01 * x4
            + 1.75 * x5
            + 0.00001 * x6
            + 2.73 * x7
        )

        f2 = (
            (
                1.16
                - 0.3717 * x2 * x4
                - 0.00931 * x2 * x10
                - 0.484 * x3 * x9
                + 0.01343 * x6 * x10
            )
            / 1.0
        ).clip(0.0)

        f3 = (
            (
                0.261
                - 0.0159 * x1 * x2
                - 0.188 * x1 * x8
                - 0.019 * x2 * x7
                + 0.0144 * x3 * x5
                + 0.87570001 * x5 * x10
                + 0.08045 * x6 * x9
                + 0.00139 * x8 * x11
                + 0.00001575 * x10 * x11
            )
            / 0.32
        ).clip(0.0)

        f4 = (
            (
                0.214
                + 0.00817 * x5
                - 0.131 * x1 * x8
                - 0.0704 * x1 * x9
                + 0.03099 * x2 * x6
                - 0.018 * x2 * x7
                + 0.0208 * x3 * x8
                + 0.121 * x3 * x9
                - 0.00364 * x5 * x6
                + 0.0007715 * x5 * x10
                - 0.0005354 * x6 * x10
                + 0.00121 * x8 * x11
                + 0.00184 * x9 * x10
                - 0.018 * x2 * x2
            )
            / 0.32
        ).clip(0.0)

        f5 = (
            (
                0.74
                - 0.61 * x2
                - 0.163 * x3 * x8
                + 0.001232 * x3 * x10
                - 0.166 * x7 * x9
                + 0.227 * x2 * x2
            )
            / 0.32
        ).clip(0.0)

        f6 = (
            (
                (
                    28.98
                    + 3.818 * x3
                    - 4.2 * x1 * x2
                    + 0.0207 * x5 * x10
                    + 6.63 * x6 * x9
                    - 7.77 * x7 * x8
                    + 0.32 * x9 * x10
                )
                + (
                    33.86
                    + 2.95 * x3
                    + 0.1792 * x10
                    - 5.057 * x1 * x2
                    - 11 * x2 * x8
                    - 0.0215 * x5 * x10
                    - 9.98 * x7 * x8
                    + 22 * x8 * x9
                )
                + (46.36 - 9.9 * x2 - 12.9 * x1 * x8 + 0.1107 * x3 * x10)
            )
            / 3
            / 32
        ).clip(0.0)

        f7 = (
            (
                4.72
                - 0.5 * x4
                - 0.19 * x2 * x3
                - 0.0122 * x4 * x10
                + 0.009325 * x6 * x10
                + 0.000191 * x11 * x11
            )
            / 4.0
        ).clip(0.0)

        f8 = (
            (
                10.58
                - 0.674 * x1 * x2
                - 1.95 * x2 * x8
                + 0.02054 * x3 * x10
                - 0.0198 * x4 * x10
                + 0.028 * x6 * x10
            )
            / 9.9
        ).clip(0.0)

        f9 = (
            (
                16.45
                - 0.489 * x3 * x7
                - 0.843 * x5 * x6
                + 0.0432 * x9 * x10
                - 0.0556 * x9 * x11
                - 0.000786 * x11 * x11
            )
            / 15.7
        ).clip(0.0)

        return torch.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9], dim=-1)


class ResourcePlanning(MultiObjectiveTestProblem):
    r"""Water resource planning.

    Adapted problem `RE61` in https://github.com/ryojitanabe/reproblems
    """
    dim = 3
    num_objectives = 6
    _bounds = [
        (0.01, 0.45),
        (0.01, 0.1),
        (0.01, 0.1),
    ]
    _ref_point = [83500.0, 1350.0, 290000.0, 16050000.0, 355000.0, 98000.0]
    _num_original_constraints = 7

    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.
        """
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]

        f1 = 106780.37 * (x2 + x3) + 61704.67
        f2 = 3000 * x1
        f3 = 305700 * 2289 * x2 / pow(0.06 * 2289, 0.65)
        f4 = 250 * 2289 * torch.exp(-39.75 * x2 + 9.9 * x3 + 2.74)
        f5 = 25 * (1.39 / (x1 * x2) + 4940 * x3 - 80)

        # Constraint functions
        g1 = 1 - (0.00139 / (x1 * x2) + 4.94 * x3 - 0.08)
        g2 = 1 - (0.000306 / (x1 * x2) + 1.082 * x3 - 0.0986)
        g3 = 50000 - (12.307 / (x1 * x2) + 49408.24 * x3 + 4051.02)
        g4 = 16000 - (2.098 / (x1 * x2) + 8046.33 * x3 - 696.71)
        g5 = 10000 - (2.138 / (x1 * x2) + 7883.39 * x3 - 705.04)
        g6 = 2000 - (0.417 * x1 * x2 + 1721.26 * x3 - 136.54)
        g7 = 550 - (0.164 / (x1 * x2) + 631.13 * x3 - 54.48)

        g = torch.stack([g1, g2, g3, g4, g5, g6, g7], dim=-1)
        g = torch.where(g < 0, -g, torch.zeros(g.size()))
        f6 = torch.sum(g, dim=-1)

        return torch.stack([f1, f2, f3, f4, f5, f6], dim=-1)
