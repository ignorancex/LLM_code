from copy import deepcopy
from itertools import combinations
from time import time

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from pandas import DataFrame, read_pickle
from poly_matrix import PolyMatrix
from pylgmath import so3op

from cert_tools import solve_sdp_cvxpy, solve_sdp_mosek
from cert_tools.admm_clique import ADMMClique
from cert_tools.base_clique import BaseClique
from cert_tools.sparse_solvers import solve_oneshot
import certifiable_uda_loc.lifted_factor_graph as lifted_factor_graph

import certifiable_uda_loc.trials as trials
from certifiable_uda_loc.settings import (
    ProblemSolutionSettings,
)
from cert_tools import HomQCQP
import certifiable_uda_loc.constraints.problem_constraint_creation as problem_constraint_creation
from pathlib import Path
import os
from typing import Tuple, Dict
from loguru import logger

# Global Defaults
ER_MIN = 1e6


def symmetrize(A: PolyMatrix):
    A_symm = PolyMatrix()
    for key1 in A.variable_dict_i:
        for key2 in A.variable_dict_j:
            if np.abs(A[key1, key2]) > 1e-10:
                A_symm[key1, key2] = A[key1, key2]
    return A_symm


class UnknownAssociationLocalizationProblem:
    """
    Localization with unknown data associations.
    Compatibility layer for now.. might migrate more thoroughly later.
    """

    def __init__(
        self,
        lifted_fg: lifted_factor_graph.LiftedFactorGraph,
        problem_settings: ProblemSolutionSettings,
    ):
        # Our layer.
        self.lifted_fg = lifted_fg
        prob_qcqp = trials.setup_problem_qcqp(
            self.lifted_fg, problem_settings=problem_settings
        )
        self.cost = prob_qcqp.Q

        for lv1, A in enumerate(prob_qcqp.A_list):
            if not A.symmetric:
                prob_qcqp.A_list[lv1] = symmetrize(A)
        # Import HomQCQP to remove redundant constraints.

        prob_hom_qcqp = HomQCQP(homog_var="one_col0")
        self.Ah = PolyMatrix()
        self.Ah[prob_hom_qcqp.h, prob_hom_qcqp.h] = 1
        prob_hom_qcqp.C = prob_qcqp.Q
        prob_hom_qcqp.As = [
            A
            for A, b in zip(prob_qcqp.A_list, prob_qcqp.RHS_list)
            if np.abs(b - 1) > 0.1
        ]
        prob_hom_qcqp.var_list = list(prob_qcqp.Q.variable_dict_i.keys())
        prob_hom_qcqp.var_sizes = {key: 1 for key in prob_qcqp.Q.variable_dict_i.keys()}
        logger.info(f"\nProb Hom QCQP Amount of constraints: {len(prob_hom_qcqp.As)}")
        prob_hom_qcqp.remove_dependent_constraints()
        logger.info(
            f"\nProb Hom QCQP Amount of constraints after removing dependent: {len(prob_hom_qcqp.As)}"
        )
        self.constraints = [(A, 0.0) for A in prob_hom_qcqp.As]
        self.var_list = list(prob_qcqp.Q.variable_dict_i.keys())
        self.var_dict = prob_qcqp.Q.variable_dict_i

        self.lifted_variable_names_reordered = (
            problem_constraint_creation.get_lifted_variable_names_from_uda_group_list(
                lifted_fg.hom_var,
                lifted_fg.uda_group_list_sparse,
                lifted_fg.lifted_variables_names_all,
            )
        )

    def get_locking_constraint(self, index):
        """Get constraint that locks a particular pose to its ground truth value
        rather than adding a prior cost term. This should remove the gauge freedom
        from the problem, giving a rank-1 solution"""
        # UTIAS did it so it must have helped :)
        r_gt = self.R_gt[index].reshape((9, 1), order="F")
        constraints = []
        for k in range(9):
            A = PolyMatrix()
            e_k = np.zeros((1, 9))
            e_k[0, k] = 1
            A["h", index] = e_k / 2
            A["h", "h"] = -r_gt[k]
            constraints += [(A, 0.0)]
        return constraints

    def solve_sdp(
        self,
        options_cvxpy: Dict,
        solver="mosek",
        primal=False,
        adjust=True,
    ) -> Tuple[np.ndarray, Dict]:
        """Solve non-chordal SDP for PGO problem without using ADMM
        solver: either "mosek" or "cvxpy"
        """
        # Convert to sparse matrix from polymatrix
        logger.info(
            f"Solving SDP. Solver: {solver}, primal: {primal}, adjust: {adjust}"
        )
        Cost = self.cost.get_matrix(self.var_list)
        Constraints = [(A.get_matrix(self.var_dict), b) for A, b in self.constraints]
        Constraints += [(self.Ah.get_matrix(self.var_dict), 1)]
        if solver == "mosek":
            X, info = solve_sdp_mosek(
                Q=Cost,
                Constraints=Constraints,
                primal=primal,
                adjust=True,
                verbose=True,
                options=options_cvxpy,
            )
        if solver == "cvxpy":
            X, info = solve_sdp_cvxpy(
                Q=Cost,
                Constraints=Constraints,
                primal=primal,
                adjust=adjust,
                verbose=True,
                options=options_cvxpy,
            )

        return X, info

    def plot_matrices(self, folder: str = None, ordered_vars=None):

        if ordered_vars is None:
            ordered_vars = list(self.cost.variable_dict_i.keys())

        Path(folder).mkdir(parents=True, exist_ok=True)

        Cost = self.cost.get_matrix(self.var_dict)
        plt.matshow(Cost.todense())
        if folder is not None:
            plt.savefig(os.path.join(folder, "cost.png"))
        A_all = np.sum([A.get_matrix(self.var_dict) for A, b in self.constraints])
        plt.figure()
        plt.matshow(A_all.todense())
        if folder is not None:
            plt.savefig(os.path.join(folder, "A_all.png"))

        A_all_poly = self.constraints[0][0]
        for lv1 in range(1, len(self.constraints)):
            A_all_poly = A_all_poly + self.constraints[lv1][0]
        fig, ax, im = A_all_poly.spy(
            variables_i=ordered_vars,
            variables_j=ordered_vars,
        )
        if folder is not None:
            fig.savefig(os.path.join(folder, "A_all_sparsity.png"))
