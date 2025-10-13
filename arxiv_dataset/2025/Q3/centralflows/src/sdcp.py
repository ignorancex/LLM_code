from dataclasses import dataclass

import cvxpy as cp
import numpy as np


"""Core code for solving Semidefinite Complementarity Problems (SDCPs) that
   arise when discretizing central flows."""

@dataclass
class SDCPSolver:
    """
    Solver for Semidefinite Complementary Problems SDCP(A,B):
        A + B[X] ≥ 0,  ⟨X, A+B[X]⟩ = 0,  X ≥ 0
    Where:
        - A is a k x k symmetric matrix
        - B is a k x k x k x k PSD tensor operator on matrices
        - X is a k x k PSD matrix variable
    """

    def __post_init__(self):
        """Initialize empty dictionary to store compiled problems."""
        self.problems = dict()

    def create_problem(self, k):
        """
        Creates a SDCP of size k which will be compiled and reused on subsequent calls

        Args:
            k (int): Size of the problem (matrices will be k x k)
        """

        A = cp.Parameter((k, k), name="A", symmetric=True)
        L = cp.Parameter((k**2, k**2), name="L")  # B = LLᵀ (cholesky square root)
        X = cp.Variable((k, k), name="X", PSD=True)
        # quadratic program: min_X  1/2 B[X,X] + ⟨A,X⟩
        # equivalent to the SDCP when B is PSD
        objective = 0.5 * cp.sum_squares(L.T @ cp.vec(X)) + cp.sum(cp.multiply(A, X))
        prob = cp.Problem(cp.Minimize(objective))
        self.problems[k] = prob

    def __call__(self, A, B):
        """
        Args:
            A (numpy.ndarray): k x k symmetric matrix
            B (numpy.ndarray): k x k x k x k PSD tensor operator

        Returns:
            numpy.ndarray: k x k PSD matrix X solving SDCP(A,B)
        """
        k = A.shape[0]
        if k == 0:
            return np.zeros((0, 0))
        A, B = np.array(A, dtype=np.float64), np.array(B, dtype=np.float64)
        A_evals, A_evecs = np.linalg.eigh(A)
        if (A_evals).min() >= 0:
            return np.zeros_like(A)

        # normalize B to have std 1 for numerical stability
        # we will undo this at the end
        B_size = np.sqrt(np.mean(B**2) + 1e-8)
        B = B.reshape(k**2, k**2)
        B = B / B_size

        # safely construct L, which satisfies B = LLᵀ
        B = (B + B.T) / 2
        evals, evecs = np.linalg.eigh(B)
        evals = np.maximum(evals, 0)
        L = evecs @ np.diag(np.sqrt(evals)) @ evecs.T

        # compile the SDCP of size k if it doesn't exist
        if k not in self.problems:
            self.create_problem(k)

        # pass the parameters to the SDCP and solve
        cone_problem = self.problems[k]
        params = cone_problem.param_dict
        params["A"].value = A
        params["L"].value = L
        cone_problem.solve(
            solver="CVXOPT",
            abstol=1e-11,
        )
        X = cone_problem.var_dict["X"].value

        return X / B_size
    