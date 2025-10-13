import warnings
from typing import Self

import cvxpy as cp
import numpy as np


class Polytope:
    """
    Polytope: a single convex polytope internally stored in H-representation A x <= b
    where A is an k x n matrix and b and k x 1 column matrix.
    x is a vector of n coordinates in E^n (Euclidean n-space).
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, n: int):
        assert isinstance(A, np.ndarray)
        assert isinstance(b, np.ndarray)
        assert isinstance(n, int)
        assert A.shape[0] == b.shape[0]
        assert A.shape[1] == n
        assert b.shape[1] == 1

        # Constraint dimension
        self.n = n

        # Number of constraints
        self.num_constr = A.shape[0]

        # H-representation A x <= b
        self.A = A
        self.b = b

        # cache projection problem
        self.project = Argmin(self)

    @classmethod
    def full_space(cls, n: int) -> Self:
        assert isinstance(n, int)
        return Polytope(A=np.zeros((1, n)), b=np.ones((1, 1)), n=n)

    def intersect_with(self, other: Self) -> Self:
        assert self.n == other.n
        return Polytope(A=np.vstack((self.A, other.A)), b=np.vstack((self.b, other.b)), n=self.n)

    def c_x(self, x: np.ndarray) -> float:
        """
        return LHS of constraint inequality
        c_x(x) <= 0

        :param np.array x: input vector (n,1)
        :return float: c_x(x)
        """
        assert np.shape(x) == (self.n, 1)
        return self.A @ x - self.b

    def dx_c_x(self) -> np.ndarray:
        """
        return gradient of LHS of constraint inequality
        c_x(x) <= 0

        :return np.ndarray: dx_c_x(x)
        """
        return self.A

    def proj_2(self, z: np.ndarray):
        """
        return closest interior point of constraint set with respect to 2-norm

        :param np.array z: input vector (n,1)
        :return np.array: projection x (n,1)
        """
        assert np.shape(z) == (self.n, 1)

        quad = np.eye(self.n)
        lin = -2 * z

        return self.project.solve(quad, lin, verify_psd=False)


class Argmin:
    def __init__(self, constraints: Polytope):
        assert isinstance(constraints, Polytope)
        self.n = constraints.n

        # variables
        self.x = cp.Variable(shape=(self.n, 1), name="x")

        # parameters
        self.quad = cp.Parameter(shape=(self.n, self.n), name="quad", symmetric=True)
        self.lin = cp.Parameter(shape=(self.n, 1), name="lin")

        # objective
        objective = cp.Minimize(
            cp.quad_form(self.x, self.quad, assume_PSD=True) + self.lin.T @ self.x
        )

        # constraints
        constraints = [constraints.A @ self.x <= constraints.b]

        # problem
        self.problem = cp.Problem(objective, constraints)

    def solve(
        self, quad: np.ndarray, lin: np.ndarray, verify_psd: bool = False
    ) -> np.ndarray | None:
        assert np.shape(quad) == (self.n, self.n)
        assert np.shape(lin) == (self.n, 1)

        if verify_psd:
            assert np.all(np.linalg.eigvals(quad) >= 0)

        # verify symmetric
        assert np.allclose(quad, quad.T)

        self.quad.value = quad
        self.lin.value = lin

        with warnings.catch_warnings(action="ignore"):
            self.problem.solve(solver=cp.SCS, warm_start=True, ignore_dpp=True, eps=1e-8)

        if self.problem.status != "optimal":
            warnings.warn("no qp solution found!")
            return None

        return self.x.value
