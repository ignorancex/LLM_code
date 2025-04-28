# wlstsq.py
"""Solver class for the weighted least-squares problem."""

__all__ = [
    "WeightedLSTSQSolver",
]

import numpy as np

import opinf


class WeightedLSTSQSolver:
    r"""Solver for a weighted least-squares problem (or problems).

    ..math:
        \min_{\hat{\mathbf{o}}}_{i}} \left\|
            \mathbf{D}\hat{\mathbf{o}}_{i}
            - \hat{\mathbf{y}}_{i}
        \right\|_{\mathbf{R}^{rr}_{i}}^{2}
        + \left\| \boldsymbol{\Gamma}\hat{\mathbf{o}}_{i} \right\|_{2}^{2}

    for :math:`i = 1, \ldots, r`. In this class, the regularizer
    :math:`\boldsymbol{\Gamma}` is the same for each :math:`i`.
    """

    _METHODS = (
        "svd",
        "lstsq",
        "normal",
    )

    def __init__(
        self,
        weights: np.ndarray,
        regularizer: float = 0.0,
        method: str = "lstsq",
    ):
        """Store the regularizer and initialize attributes.

        Parameters
        ----------
        weights : (r, m, m) or (m, m) ndarray
            Collection of r positive definite matrices defining the weighted
            norms for each problem. Specifically, for a (k,) ndarry x,
            ||x||_i = ||weights[i] @ x||_{2}.
            In the notes, weights[i] is sqrt(R_{i}^{rr}).
            If a two-dimensional array, it is assumed that r = 1.
        regularizer : (d, d) or (d,) ndarray or float.
            Regularization hyperparameters.
            * Symmetric semi-positive-definite regularization matrix P
            * The diagonal entries of P.
            * A float, in which case P = regularizer * I.
        method : str
            The strategy for solving the regularized least-squares problem.
            * "lstsq": take the SVD of the stacked data matrix [A.T | P.T].T.
            * "normal": solve the normal equations (A.T R A + P.T P) X = A.T B.
            This only applies if ``regularizer`` is a matrix.
        """
        self.__solvers = []

        self.weights = weights
        self.regularizer = regularizer
        self.method = method

    @property
    def solvers(self):
        """list of opinf.lstsq objects : Underlying least-squares engines for
        each independent least-squares problem.
        """
        return self.__solvers

    # Properties: hyperparameters ---------------------------------------------
    @property
    def weights(self):
        """(r, k, k) ndarray:
        Collection of r positive definite matrices defining the weighted
        norms for each problem. Specifically, for a (m,) ndarry x,

            ||x||_i = ||weights[i] @ x||_{2}.

        In the notes, weights[i] is sqrt(R_{i}^{rr}).
        """
        return self.__weights

    @weights.setter
    def weights(self, Rs):
        """Set the weights, determining dimensions r and m."""
        if not isinstance(Rs, np.ndarray):
            Rs = np.array(Rs)
        if Rs.ndim == 2:
            Rs = Rs.reshape((1, Rs.shape[0], Rs.shape[1]))
        if Rs.ndim != 3 or Rs.shape[1] != Rs.shape[2]:
            raise ValueError("expected (r, m, m) ndarray of weights")
        self.__r = Rs.shape[0]
        self.__m = Rs.shape[1]
        self.__weights = Rs

    @property
    def regularizer(self):
        """float or (d, d) ndarray : regularizer, either a symmetric
        semi-positive-definite matrix P or a scalar, in which case
        P = regularizer * Identity.
        """
        return self.__reg

    @regularizer.setter
    def regularizer(self, P):
        """Set the regularizer."""
        self.__reg = P
        for solver in self.__solvers:
            solver.regularizer = P

    @property
    def method(self):
        """str : Strategy for solving the regularized least-squares problem.
        * "svd": take the SVD of the stacked data matrix [(sqrt(R)A).T|P.T].T.
        * "normal": solve the normal equations (A.T R A + P.T P) X = A.T R B.
        """
        return self.__method

    @method.setter
    def method(self, method):
        """Set the method."""
        if method not in self._METHODS:
            mtdlist = ", ".join([f"'{mtd}'" for mtd in self._METHODS])
            raise ValueError(f"method must be one of {mtdlist}")
        if method == "lstsq":
            method = "svd"

        self.__method = method
        for solver in self.__solvers:
            solver.method = method

    # Properties: dimensions --------------------------------------------------
    @property
    def r(self):
        """int > 0 : number of independent least-squares problems to solve."""
        return self.__r

    @property
    def m(self):
        """int > 0 : number of data instances in the least-squares problem."""
        return self.__m

    @property
    def d(self):
        """int > 0 : number of unknowns in each least-squares problem."""
        return self.__d

    # Main methods ------------------------------------------------------------
    def _check_is_trained(self):
        if not self.__solvers:
            raise AttributeError("solver not trained (call fit())")

    def fit(self, lhs, rhs):
        """Store the data matrices defining the least-squares problems
        and initialized solvers for each independent sub-problem.

        Parameters
        ----------
        lhs : (m, d) ndarray
            Unweighted left-hand side data matrix (D in the notes).
        rhs : (r, m) or (m,) ndarray
            Unweighted right-hand side data matrix [z1 | z2 | ... zr]^T.
            May be one-dimensional if r = 1.
        """
        # Check dimensions.
        if lhs.shape != (_shape := (self.m, lhs.shape[1])):
            raise ValueError(f"expected lhs.shape == {_shape}")
        if np.ndim(rhs) == 1:
            rhs = np.reshape(rhs, (1, -1))
        if rhs.shape != (_shape := (self.r, self.m)):
            raise ValueError(f"expected rhs.shape == {_shape}")
        self.__d = lhs.shape[1]

        # Initialize underlying solvers.
        if np.isscalar(self.regularizer):
            SolverClass = opinf.lstsq.L2Solver
        else:
            SolverClass = opinf.lstsq.TikhonovSolver

        self.__solvers = [
            SolverClass(self.regularizer).fit(
                self.weights[i] @ lhs, self.weights[i] @ rhs[i]
            )
            for i in range(self.r)
        ]

        # Set the solver method (only for Tikhonov solvers).
        if SolverClass is opinf.lstsq.TikhonovSolver:
            for solver in self.__solvers:
                solver.method = self.method

        return self

    def solve(self):
        """Solve each underlying least-squares problem.

        Returns
        -------
        Ohat : (r, d) or (d,) ndarray
            Least-squares solution Ohat = [ ohat_1 | ... | ohat_r ]^T; each
            column is the solution to one subproblem. Flattened if r == 1.
        """
        self._check_is_trained()

        Ohat = np.concatenate([solver.solve() for solver in self.__solvers])
        if Ohat.shape != (_shape := (self.r, self.d)):
            raise RuntimeError(f"Ohat.shape != {_shape}")

        return Ohat[0] if self.r == 1 else Ohat


class WeightedLSTSQSolverMulti(WeightedLSTSQSolver):
    r"""Solver for a weighted least-squares problem (or problems).

    ..math:
        \min_{\hat{\mathbf{o}}}_{i}} \left\|
            \mathbf{D}\hat{\mathbf{o}}_{i}
            - \hat{\mathbf{y}}_{i}
        \right\|_{\mathbf{R}^{rr}_{i}}^{2}
        + \left\| \boldsymbol{\Gamma}_{i}\hat{\mathbf{o}}_{i} \right\|_{2}^{2}

    for :math:`i = 1, \ldots, r`. In this class, the regularizer
    :math:`\boldsymbol{\Gamma}_{i}` is DIFFERENT for each :math:`i`.
    """

    def __init__(
        self,
        weights: np.ndarray,
        regularizer: float = 0.0,
        method: str = "lstsq",
    ):
        """Store the regularizer and initialize attributes.

        Parameters
        ----------
        weights : (r, m, m) or (m, m) ndarray
            Collection of r positive definite matrices defining the weighted
            norms for each problem. Specifically, for a (k,) ndarry x,
            ||x||_i = ||weights[i] @ x||_{2}.
            In the notes, weights[i] is sqrt(R_{i}^{rr}).
            If a two-dimensional array, it is assumed that r = 1.
        regularizer : List of r {(d, d) or (d,) ndarray or float}.
            Regularization hyperparameters. Each entry is one of
            * Symmetric semi-positive-definite regularization matrix P
            * The diagonal entries of P.
            * A float, in which case P = regularizer * I.
        method : str
            The strategy for solving the regularized least-squares problem.
            * "lstsq": take the SVD of the stacked data matrix [A.T | P.T].T.
            * "normal": solve the normal equations (A.T R A + P.T P) X = A.T B.
            This only applies if ``regularizer`` is a matrix.
        """
        WeightedLSTSQSolverMulti.__init__(self, weights, regularizer, method)

    @WeightedLSTSQSolver.regularizer.setter
    def regularizer(self, Ps):
        """Set the regularizers."""
        if Ps is None:
            Ps = np.zeros(self.r)
        if (nPs := len(Ps)) != self.r:
            raise ValueError(f"len(regularizer) = {nPs} != {self.r} = r")

        self._WeightedLSTSQSolver__reg = Ps
        for solver, P in zip(self.solvers, Ps):
            solver.regularizer = P

    def fit(self, lhs, rhs):
        """Store the data matrices defining the least-squares problems
        and initialized solvers for each independent sub-problem.

        Parameters
        ----------
        lhs : (m, d) ndarray
            Unweighted left-hand side data matrix (D in the notes).
        rhs : (r, m) or (m,) ndarray
            Unweighted right-hand side data matrix [z1 | z2 | ... zr]^T.
            May be one-dimensional if r = 1.
        """
        # Check dimensions.
        if lhs.shape != (_shape := (self.m, lhs.shape[1])):
            raise ValueError(f"expected lhs.shape == {_shape}")
        if np.ndim(rhs) == 1:
            rhs = np.reshape(rhs, (1, -1))
        if rhs.shape != (_shape := (self.r, self.m)):
            raise ValueError(f"expected rhs.shape == {_shape}")
        self._WeightedLSTSQSolver__d = lhs.shape[1]

        # Initialize underlying solvers.
        self._WeightedLSTSQSolver__solvers = []
        for weight, rhsi, reg in zip(self.weights, rhs, self.regularizer):
            SolverClass = opinf.lstsq.TikhonovSolver
            if np.isscalar(reg):
                SolverClass = opinf.lstsq.L2Solver

            newsolver = SolverClass(reg).fit(weight @ lhs, weight @ rhsi)
            if hasattr(newsolver, "method"):
                newsolver.method = self.method

            self.solvers.append(newsolver)

        return self
