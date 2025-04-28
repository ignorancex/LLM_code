# step3_estimate.py
"""Estimate the system parameters with GP-powered OpInf."""

__all__ = [
    "estimate_posterior",
]

import logging
import warnings
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt

import config
import bayes
import wlstsq

import opinf


__MAXOPTVAL = 1e12  # Ceiling for optimization.
__DEFAULT_SEARCH_GRID = np.logspace(-16, 4, 81)  # Search grid.


def _posterior_autoregularized_multisample(
    regularizer_grid: np.ndarray,
    time_domain: np.ndarray,
    time_domain_estimated: np.ndarray,
    snapshots_estimated: np.ndarray,
    num_samples: int,
    lstsq_solver: wlstsq.WeightedLSTSQSolver,
    rom: opinf.models.ContinuousModel,
) -> bayes.BayesianROM:
    r"""Use an error-based optimization to select an appropriate regularization
    hyperparamter for the operator inference regression.

    .. math::
       \ohat_i = (D^T W_i D + \lambda_i I)^{-1} D^T W_i ddts_i

    Use ``num_samples`` posterior draws to check that the posterior gives
    stable solutions.

    Parameters
    ----------
    regularizer_grid : (num_regs,) ndarray
        Grid of regularization values to try (followed by an optimization).
    time_domain : (k,) ndarray
        Time domain over which to solve the model for a stability check.
    time_domain_estimated : (m',) ndarray
        Time domain corresponding to the GP estimates of the snapshots.
    snapshots_estimated : (r, m') ndarray
        GP state estimates of the available training snapshots.
    num_samples : int
        Number of posterior draws to do for the stability check.
    lstsq_solvers : list of wlstsq.WeightedLSTSQSolver objects
        Solvers for the least-squares problem(s) (already 'fit' to the data).
    rom : opinf.models.ContinuousModel
        Reduced-order model object (**not** fit to the data)

    Returns
    -------
    Bayesian model.
    """
    shift = np.mean(snapshots_estimated, axis=1).reshape((-1, 1))
    limits = 5 * np.abs(snapshots_estimated - shift).max(axis=1)
    snapshotnorm = la.norm(snapshots_estimated)
    initial_conditions = snapshots_estimated[:, 0]

    def unstable(_solution, size):
        """Return True if the solution is unstable."""
        if _solution.shape[-1] != size:
            return True
        return np.any(np.abs(_solution - shift).max(axis=1) > limits)

    def get_bayesian_model(reg):
        """Form and solve the regression for the given regularization value."""
        # Posterior mean.
        lstsq_solver.regularizer = reg
        means = np.atleast_2d(lstsq_solver.solve())
        rom._extract_operators(means)

        # Posterior covariance.
        precisions = []
        reg2 = lstsq_solver.regularizer**2
        for subsolver, mean in zip(lstsq_solver.solvers, means):
            # Raw precision matrix.
            RsqrtD = subsolver.data_matrix  # = Rsqrt @ D
            invSigma = (RsqrtD.T @ RsqrtD) + (reg2 * np.eye(mean.size))
            precisions.append(invSigma)
        try:
            return bayes.BayesianROM(means, precisions, rom)
        except np.linalg.LinAlgError as ex:
            if ex.args[0] == "Matrix is not positive definite":
                return None
            raise

    def _training_error(logreg):
        """Get the solution for a single regularization candidate."""
        opinf_regularizer = 10**logreg
        print(
            f"Testing regularizer {opinf_regularizer:.4e}...",
            end="",
            flush=True,
        )
        bayesian_model = get_bayesian_model(opinf_regularizer)
        if bayesian_model is None:
            print("Covariance not SPD")
            return __MAXOPTVAL

        # Sample the posterior distribution and check for stability.
        draws = []
        for _ in range(num_samples):
            for tmdmn in (time_domain, time_domain_estimated):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    draw = bayesian_model.predict(
                        initial_conditions=initial_conditions,
                        timepoints=tmdmn,
                        input_func=rom.input_func,
                    )
                if unstable(draw, tmdmn.size):
                    print("UNSTABLE")
                    return __MAXOPTVAL
            draws.append(draw)

        rom_solution = np.mean(draws, axis=0)
        error = la.norm(rom_solution - snapshots_estimated) / snapshotnorm
        print(f"{error:.2%} error")
        return error

    # Test each regularization hyperparameter.
    regularizer_grid = np.atleast_1d(regularizer_grid)
    if (num_tests := len(regularizer_grid)) == 1:
        search_bounds = [regularizer_grid[0] / 10, 10 * regularizer_grid[0]]
    else:
        # GRID SEARCH.
        _smallest_error, _best_reg_index = __MAXOPTVAL, None
        regularizer_grid = np.sort(regularizer_grid)
        print("\nGRIDSEARCH")
        for i, reg in enumerate(regularizer_grid):
            print(f"({i+1:d}/{num_tests:d}) ", end="")
            if (error := _training_error(np.log10(reg))) < _smallest_error:
                _smallest_error = error
                _best_reg_index = i
        if _best_reg_index is None:
            raise ValueError("grid search failed!")
        best_reg = regularizer_grid[_best_reg_index]

        if _best_reg_index == 0:
            print("\nWARNING: extend regularizer_grid to the left!")
            search_bounds = [best_reg / 100, regularizer_grid[1]]
        elif _best_reg_index == num_tests - 1:
            print("\nWARNING: extend regularizer_grid to the right!")
            search_bounds = [regularizer_grid[-2], 100 * best_reg]
        else:
            search_bounds = [
                regularizer_grid[_best_reg_index - 1],
                regularizer_grid[_best_reg_index + 1],
            ]

        message = f"Best regularization via gridsearch: {best_reg:.4e}"
        print(message + "\n")
        logging.info(message)

    # Follow up grid search with minimization-based search.
    print("1D OPTIMIZATION")
    opt_result = opt.minimize_scalar(
        _training_error, method="bounded", bounds=np.log10(search_bounds)
    )

    if opt_result.success and opt_result.fun != __MAXOPTVAL:
        regularizer = 10**opt_result.x
        message = f"Best regularization via optimization: {regularizer:.4e}"
        print(message)
        logging.info(message)
    else:
        regularizer = best_reg
        print("Optimization failed, falling back on gridsearch")

    return get_bayesian_model(regularizer)


def estimate_posterior(
    time_domain: np.ndarray,
    gps: list,
    inputs: np.ndarray,
) -> bayes.BayesianROM:
    """Construct the posterior parameter distribution.

    Parameters
    ----------
    time_domain : (k,) ndarray
        Time domain over which to solve the model.
    gps : list of trained gpkernel.GP_RBFW objects.
        Gaussian processes for each state variable, already fit to data.
    inputs : (k,) ndarray or None
        Inputs corresponding to the GP state estimates (if present).

    Returns
    -------
    bayes.BayesianROM
        Initialized Bayesian model.
    """
    with opinf.utils.TimedBlock("constructing posterior hyperparameters\n"):
        rom = config.ReducedOrderModel()
        rom.state_dimension = len(gps)

        # Construct the data matrix RHS ddts vector, and weight matrix.
        state_estimates = np.array([gp.state_estimate for gp in gps])
        D = rom._assemble_data_matrix(state_estimates, inputs)
        rhs = np.array([gp.ddt_estimate for gp in gps])
        Rsqrts = np.array([gp.sqrtW for gp in gps])

        # Pepare a solver and reduced-order model.
        lstsq_solver = wlstsq.WeightedLSTSQSolver(Rsqrts, 1)
        lstsq_solver.fit(D, rhs)

        # Choose the regularizer automatically through an optimization.
        return _posterior_autoregularized_multisample(
            regularizer_grid=__DEFAULT_SEARCH_GRID,
            time_domain=time_domain,
            time_domain_estimated=gps[0].t_estimation,
            snapshots_estimated=state_estimates,
            num_samples=20,
            lstsq_solver=lstsq_solver,
            rom=rom,
        )
