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
import wlstsq
import bayes

import opinf


__MAXOPTVAL = 1e12  # Ceiling for optimization.
__DEFAULT_SEARCH_GRID = np.logspace(-16, 4, 81)  # Search grid.


def _posterior_autoregularized_multisample(
    regularizer_grid: np.ndarray,
    prediction_time_domain: np.ndarray,
    initial_conditions: list[np.ndarray],
    training_time_domain: np.ndarray,
    training_snapshots: list[np.ndarray],
    num_draws: int,
    lstsq_solver: wlstsq.WeightedLSTSQSolver,
    rom: opinf.models.ContinuousModel,
) -> bayes.BayesianROM:
    r"""Use an error-based optimization to select an appropriate regularization
    hyperparameter for the operator inference regression.

    That is, find an appropriate ``reg`` for solving the problem

        \ohat = (sum_l D[l]^T W[l] D[l] + reg I)^{-1}
                (sum_l D[l]^T W[l] ddts[l])

    or the equivalent least-squares regression

        \ohat = argmin[µ] sum_l ||D[l] µ - ddts[l]||_{W[l]}^2 + reg ||µ||_2^2

                          || [ D[1] ]       [ ddts[1] ] ||^2
                          || [ D[2] ]       [ ddts[2] ] ||
              = argmin[µ] || [  :   ] µ  -  [   :     ] ||    + reg ||µ||_2^2,
                          || [ D[L] ]       [ ddts[L] ] ||_W

    where W = block_diag(W[1], W[2], ..., W[L]).

    Use ``num_draws`` posterior draws to check that the posterior gives
    stable solutions.

    Parameters
    ----------
    regularizer_grid : (num_regs,) ndarray
        Grid of regularization values to try (followed by an optimization).
    prediction_time_domain : (k,) ndarray
        Time domain over which to solve the model for a stability check.
    initial_conditions : list of L (r,) ndarray
        Initial conditions for the ROM, one for each trajectory.
    training_time_domain : (m,) ndarray
        Time domain corresponding to the GP estimates of the snapshots.
    training_snapshots : list of L (r, m) ndarrays
        GP state estimates of the available training snapshots.
    num_draws : int
        Number of posterior draws to do for the stability check.
    lstsq_solvers : list of wlstsq.WeightedLSTSQSolver objects
        Solvers for the least-squares problem(s) (already 'fit' to the data).
    rom : opinf.models.ContinuousModel
        Reduced-order model object (**not** yet fit to the data).
    quad_regularizer : bool
        If ``True``, only regularize the quadratic terms in the model.

    Returns
    -------
    Bayesian reduced-order model trained with the optimal regularization.
    """
    shifts = [np.mean(Q, axis=1).reshape((-1, 1)) for Q in training_snapshots]
    limits = [
        5 * np.abs(Q - qbar).max(axis=1)
        for Q, qbar in zip(training_snapshots, shifts)
    ]
    snapshot_norms = [la.norm(Q) for Q in training_snapshots]

    def unstable(_solution, i, size):
        """Return ``True`` if the solution is unstable."""
        if _solution.shape[-1] != size:
            return True
        return np.any(np.abs(_solution - shifts[i]).max(axis=1) > limits[i])

    def get_bayesian_model(reg):
        """Form and solve the regression for the given regularization value."""
        # Posterior mean.
        lstsq_solver.regularizer = reg
        means = np.atleast_2d(lstsq_solver.solve())
        rom._extract_operators(means)

        # Posterior covariance.
        precisions = []
        reg2 = lstsq_solver.regularizer**2
        for solver, mean in zip(lstsq_solver.solvers, means):
            reg2 = solver.regularizer**2 * np.eye(mean.size)
            # Raw precision matrix.
            RsqrtD = solver.data_matrix  # = Rsqrt @ D
            invSigma = (RsqrtD.T @ RsqrtD) + reg2
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
            return __MAXOPTVAL

        error = 0
        for ell, params in enumerate(config.input_parameters):
            input_func = config.input_func_factory(params)
            ICs = initial_conditions[ell]
            draws = []
            for _ in range(num_draws):
                for tmdmn in (prediction_time_domain, training_time_domain):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        draw = bayesian_model.predict(
                            initial_conditions=ICs,
                            timepoints=tmdmn,
                            input_func=input_func,
                        )
                    if unstable(draw, ell, tmdmn.size):
                        print("UNSTABLE")
                        return __MAXOPTVAL
                draws.append(draw)

            rom_solution = np.mean(draws, axis=0)
            error += (
                la.norm(rom_solution - training_snapshots[ell])
                / snapshot_norms[ell]
            )
        error /= len(config.input_parameters)
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

        return get_bayesian_model(regularizer)

    raise RuntimeError("Regularization search optimization FAILED")


def estimate_posterior(
    prediction_time_domain: np.ndarray,
    gps: list,
    training_inputs: np.ndarray,
    initial_conditions: list = None,
) -> bayes.BayesianROM:
    r"""Construct the posterior parameter distribution.

    Parameters
    ----------
    prediction_time_domain : (k,) ndarray
        Time domain over which to solve the model.
    gps : list of L lists of num_modes gpkernel.GP_RBFW objects.
        Gaussian processes for each state variable, already fit to data.
        gps[ell][i] is the GP for the i^th mode of the ell^th trajectory.
    training_inputs : list of (M, k) ndarrays
        Inputs corresponding to the state estimates.
    initial_conditions : list of L (r,) ndarrays
        Initial conditions for each trajectory.
        Only used / required if ``opinf_regularizer=None``.

    Returns
    -------
    bayes.BayesianROM
        Initialized Bayesian model.
    """
    with opinf.utils.TimedBlock("constructing posterior hyperparameters\n"):
        rom = config.ReducedOrderModel()
        rom.state_dimension = len(gps[0])

        # Get GP state estimates and construct the data matrix.
        data_matrix, training_snapshots = [], []
        for ell in range(len(config.input_parameters)):
            state_estimates = np.array([gp.state_estimate for gp in gps[ell]])
            training_snapshots.append(state_estimates)
            data_matrix.append(
                rom._assemble_data_matrix(
                    state_estimates,
                    training_inputs[ell],
                )
            )
        data_matrix = np.vstack(data_matrix)

        # For each mode, get the time derivative data and the weight matrix.
        rhs, weights = [], []
        for i in range(len(gps[0])):
            rhs.append(
                np.concatenate([gps_ell[i].ddt_estimate for gps_ell in gps])
            )
            weights.append(
                la.block_diag(*[gps_ell[i].sqrtW for gps_ell in gps])
            )
        rhs = np.array(rhs)

        # Choose the regularizer automatically through an optimization.
        lstsq_solver = wlstsq.WeightedLSTSQSolver(weights, 1)
        lstsq_solver.fit(data_matrix, rhs)
        return _posterior_autoregularized_multisample(
            regularizer_grid=__DEFAULT_SEARCH_GRID,
            prediction_time_domain=prediction_time_domain,
            initial_conditions=initial_conditions,
            training_time_domain=gps[0][0].t_estimation,
            training_snapshots=training_snapshots,
            num_draws=20,
            lstsq_solver=lstsq_solver,
            rom=rom,
        )
