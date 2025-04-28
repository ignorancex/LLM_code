# step3_estimate.py
"""Estimate system parameters with GP-powered OpInf."""

__all__ = [
    "estimate_posterior",
]

import logging
import warnings
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt

import opinf

import bayes
import config
import wlstsq


__MAXOPTVAL = 1e12  # Ceiling for optimization.
__DEFAULT_SEARCH_GRID = np.logspace(-16, 5, 22)  # Search grid.


def _posterior_autoregularized_multisample(
    regularizer_grid: np.ndarray,
    time_domain_prediction: np.ndarray,
    time_domain_estimated: np.ndarray,
    snapshots_estimated: np.ndarray,
    initial_conditions: np.ndarray,
    num_samples: int,
    lstsq_solver: wlstsq.WeightedLSTSQSolver,
    model: config.Model,
) -> bayes.BayesianODE:
    r"""Use an error-based optimization to select an appropriate regularization
    hyperparamter for the parameter estimation regression.

        \ohat = (D^T W D + \lambda I)^{-1} D^T W ddts

    Use ``num_samples`` posterior draws to check that the posterior gives
    stable solutions.

    Parameters
    ----------
    regularizer_grid : (num_regs,) ndarray
        Grid of regularization values to try (followed by an optimization).
    time_domain_prediction : (k,) ndarray
        Time domain over which to solve the model for a stability check.
    initial_conditions : (r,) ndarray
        Initial conditions for the model.
    time_domain_estimated : (m',) ndarray
        Time domain corresponding to the GP estimates of the snapshots.
    snapshots_estimated : (r, m') ndarray
        GP state estimates of the available training snapshots.
    num_samples : int
        Number of posterior draws to do for the stability check.
    lstsq_solver : wlstsq.WeightedLSTSQSolver
        Solver for the least-squares problem (already 'fit' to the data).
    model : config.Model
        Model object for running simulations.

    Returns
    -------
    bayes.BayesianODE
        Bayesian ODE model.
    """
    shift = np.mean(snapshots_estimated, axis=1).reshape((-1, 1))
    limits = 5 * np.abs(snapshots_estimated - shift).max(axis=1)
    snapshotnorm = la.norm(snapshots_estimated)
    if initial_conditions is None:
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
        mean = lstsq_solver.solve()
        model.parameters = mean

        # Posterior precision matrix.
        sqrtW_D = lstsq_solver.solvers[0].data_matrix  # = sqrt(W) @ D
        precision = (sqrtW_D.T @ sqrtW_D) + (reg**2 * np.eye(mean.size))

        try:
            return bayes.BayesianODE(model, mean, precision)
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
            for tmdmn in (time_domain_prediction, time_domain_estimated):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    draw = bayesian_model.predict(
                        initial_conditions=initial_conditions,
                        timepoints=tmdmn,
                    )
                if unstable(draw, tmdmn.size):
                    print("UNSTABLE")
                    return __MAXOPTVAL
            draws.append(draw)

        mean_solution = np.mean(draws, axis=0)
        error = la.norm(mean_solution - snapshots_estimated) / snapshotnorm
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
    gps,
    time_domain_prediction,
    initial_conditions=None,
) -> bayes.BayesianODE:
    """Construct the posterior parameter distribution.

    Parameters
    ----------
    gps : list of trained gpkernel.GP_RBFW objects.
        Gaussian processes for each state variable, already fit to data.
    time_domain_prediction : (k,) ndarray
        Time domain over which to solve the model for stability checks.
    initial_conditions : (r,) ndarray or None
        Initial conditions for the model. If not provided, use the GP state
        estimates at the initial time.

    Returns
    -------
    bayes.BayesianODE
        Bayesian ODE model.
    """
    with opinf.utils.TimedBlock("constructing posterior hyperparameters\n"):
        model = config.Model()
        state_estimates = np.array([gp.state_estimate for gp in gps])

        # Construct the data matrix, RHS ddts vector, and weight matrix.
        D = model.data_matrix(state_estimates)
        ddt_estimates = np.concatenate([gp.ddt_estimate for gp in gps])
        W = la.block_diag(*[gp.sqrtW for gp in gps])

        # Fit a weighted least-squares solver for the problem.
        lstsq_solver = wlstsq.WeightedLSTSQSolver(W, regularizer=1)
        lstsq_solver.fit(D, ddt_estimates)

        # Select a single regularizer for all equations.
        return _posterior_autoregularized_multisample(
            regularizer_grid=__DEFAULT_SEARCH_GRID,
            time_domain_prediction=time_domain_prediction,
            time_domain_estimated=gps[0].t_estimation,
            snapshots_estimated=state_estimates,
            initial_conditions=initial_conditions,
            num_samples=20,
            lstsq_solver=lstsq_solver,
            model=model,
        )
