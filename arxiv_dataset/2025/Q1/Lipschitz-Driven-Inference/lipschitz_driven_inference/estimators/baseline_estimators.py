import numpy as np
import scipy
from scipy import stats
import gpflow
from numpy.typing import ArrayLike
from typing import List, Union
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from .estimators import (
    Estimator,
    Dataset,
    ConfidenceInterval,
)
from typing import Tuple


class SliceMeanFunction(gpflow.mean_functions.MeanFunction):
    def __init__(self, mean_function: gpflow.mean_functions.MeanFunction, dim: int):
        self.dim = dim
        self.mean_function = mean_function

    def __call__(self, X):
        Z = X[:, self.dim :]
        return self.mean_function(Z)


class OLS(Estimator):

    @staticmethod
    def name() -> str:
        return "OLS"

    def point_estimate(self, dim: int) -> float:
        """
        Compute the point estimate for the estimator.
        """
        return (
            sm.OLS(self.training_data.y[:, 0], self.training_data.X).fit().params[dim]
        )

    def _confidence_interval(
        self, dim: int, alpha: float, sandwich: bool = False
    ) -> ConfidenceInterval:
        """
        Compute the confidence interval for the OLS estimator.
        """
        if sandwich:
            cov_type = "hc1"
        else:
            cov_type = "nonrobust"
        conf_int = (
            sm.OLS(self.training_data.y[:, 0], self.training_data.X)
            .fit(cov_type=cov_type)
            .conf_int(alpha=alpha)[dim]
        )
        return ConfidenceInterval(lower=conf_int[0], upper=conf_int[1])

    def confidence_interval(self, dim: int, alpha: float = 0.05) -> ConfidenceInterval:
        return self._confidence_interval(dim, alpha, sandwich=False)

    def _sd_estimate(self, dim: int, sandwich: bool = False) -> float:
        """
        Compute the standard deviation estimate for the estimator.
        """
        if sandwich:
            cov_type = "hc1"
        else:
            cov_type = "nonrobust"
        return (
            sm.OLS(self.training_data.y[:, 0], self.training_data.X)
            .fit(cov_type=cov_type)
            .bse[dim]
        )

    def sd_estimate(self, dim: int) -> float:
        return self._sd_estimate(dim, sandwich=False)


class Sandwich(OLS):

    @staticmethod
    def name() -> str:
        return "Sandwich"

    def confidence_interval(self, dim: int, alpha: float) -> ConfidenceInterval:
        """
        Compute the confidence interval for the OLS estimator.
        """
        return self._confidence_interval(dim, alpha, sandwich=True)

    def sd_estimate(self, dim: int) -> float:
        return self._sd_estimate(dim, sandwich=True)


class KDEIW(Estimator):
    def __init__(
        self,
        training_data: Dataset,
        test_data: Dataset,
        bandwidth: Union[float, List[float]],
    ):
        super().__init__(training_data, test_data)
        self.bandwidth = bandwidth

    @staticmethod
    def name() -> str:
        return "KDEIW"

    def estimate_iw(self) -> ArrayLike:
        """
        Compute the importance weights for the estimator. This is (d P_test / d P_train)(X_train).
        """
        # Fit a Gaussian kde using sklearn on test and train data
        if isinstance(self.bandwidth, float):
            test_density_estimate = KernelDensity(bandwidth=self.bandwidth)
            train_density_estimate = KernelDensity(bandwidth=self.bandwidth)
        else:  # Run grid search to pick a bandwidth then fit the KDE on the test and train data
            test_grid = GridSearchCV(KernelDensity(), {"bandwidth": self.bandwidth})
            test_grid.fit(self.test_data.S)
            test_density_estimate = test_grid.best_estimator_
            train_grid = GridSearchCV(KernelDensity(), {"bandwidth": self.bandwidth})
            train_grid.fit(self.training_data.S)
            train_density_estimate = train_grid.best_estimator_

        test_density_estimate.fit(self.test_data.S)
        train_density_estimate.fit(self.training_data.S)
        log_density_ratio = test_density_estimate.score_samples(
            self.training_data.S
        ) - train_density_estimate.score_samples(self.training_data.S)
        return np.exp(log_density_ratio)

    def point_estimate(self, dim: int) -> float:
        """
        Compute the point estimate for the estimator.
        """
        return (
            sm.WLS(
                self.training_data.y[:, 0],
                self.training_data.X,
                weights=self.estimate_iw(),
            )
            .fit()
            .params[dim]
        )

    def confidence_interval(self, dim: int, alpha: float) -> ConfidenceInterval:
        """
        Compute the confidence interval for the OLS estimator.
        """
        conf_int = (
            sm.WLS(
                self.training_data.y[:, 0],
                self.training_data.X,
                weights=self.estimate_iw(),
            )
            .fit()
            .conf_int(alpha=alpha)[dim]
        )
        return ConfidenceInterval(lower=conf_int[0], upper=conf_int[1])

    def sd_estimate(self, dim: int) -> float:
        """
        Compute the standard deviation estimate for the estimator.
        """
        return (
            sm.WLS(
                self.training_data.y[:, 0],
                self.training_data.X,
                weights=self.estimate_iw(),
            )
            .fit()
            .bse[dim]
        )


class GLS(Estimator):
    def __init__(
        self,
        training_data: Dataset,
        test_data: Dataset,
        kernel: gpflow.kernels.Kernel,
        nugget_variance: float,
        max_likelihood: bool = True,
        srs: bool = True,
    ):
        super().__init__(training_data, test_data)
        self.Sigma, self.Sigmastar = self.init_sigma(
            kernel, nugget_variance, max_likelihood, srs
        )

    @staticmethod
    def name() -> str:
        return "GLS"

    def init_sigma(
        self,
        kernel: gpflow.kernels.Kernel,
        nugget_variance: float,
        max_likelihood: bool,
        srs: bool,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Initialize the covariance matrix Sigma.
        """
        # Compute the covariance matrix
        # Stack spatial locations and covariates, we will define the kernel to operate on spatial locations
        # And the mean function to act on covariates
        spatial_dim = self.training_data.S.shape[1]
        all_vars = np.concatenate((self.training_data.S, self.training_data.X), axis=1)
        # Define the kernel
        linear_mean = gpflow.mean_functions.Linear(
            A=1e-5 * np.ones((self.training_data.X.shape[1], 1))
        )
        gpflow.utilities.set_trainable(
            linear_mean.b, False
        )  # Xs already have a 1, don't need an intercept
        mean_function = SliceMeanFunction(linear_mean, spatial_dim)
        # Define the model
        model = gpflow.models.GPR(
            data=(all_vars, self.training_data.y),
            kernel=kernel,
            mean_function=mean_function,
            noise_variance=nugget_variance,
        )
        if max_likelihood:
            # Optimize the model
            opt = gpflow.optimizers.Scipy()
            opt.minimize(
                model.training_loss,
                model.trainable_variables,
                options=dict(maxiter=100),
            )
        all_test_vars = np.concatenate((self.test_data.S, self.test_data.X), axis=1)
        Sigma = model.kernel(all_vars, full_cov=True)
        Sigmastar = model.kernel(all_test_vars, full_cov=True)
        # Project Sigma and Sigmastar onto orthogonal complement of covariates
        if srs:
            X = self.training_data.X
            P = np.eye(self.N) - X @ np.linalg.solve(X.T @ X, X.T)
            Sigma = P @ Sigma @ P + 1e-12 * np.eye(self.N)
            Xstar = self.test_data.X
            Pstar = np.eye(self.M) - Xstar @ np.linalg.solve(Xstar.T @ Xstar, Xstar.T)
            Sigmastar = Pstar @ Sigmastar @ Pstar
        return Sigma.numpy() + model.likelihood.variance * np.eye(
            self.N
        ), Sigmastar.numpy() + model.likelihood.variance * np.eye(self.M)

    def point_estimate(self, dim: int) -> float:
        """
        Compute the point estimate for the estimator.
        """
        return (
            sm.GLS(self.training_data.y[:, 0], self.training_data.X, sigma=self.Sigma)
            .fit()
            .params[dim]
        )


    def confidence_interval(self, dim: int, alpha: float) -> ConfidenceInterval:
        """
        Compute the confidence interval for the OLS estimator.
        """
        ep = np.zeros((self.training_data.X.shape[1]))
        ep[dim] = 1
        # Compute ep X^T Sigma X ep.T

        # 1) Cholesky factorization of Sigma = L L^T
        L = np.linalg.cholesky(self.Sigma)

        # 2) Transform X and y by L^{-1} using solve
        X_tilde = np.linalg.solve(L, self.training_data.X)

        # 3) Form the "normal equations" in transformed space
        A = X_tilde.T @ X_tilde

        # 4) Compute the variance of this estimator
        beta_var = ep @ np.linalg.solve(A, ep)

        # compute the variance using the model instead
        results = sm.GLS(
            self.training_data.y[:, 0], self.training_data.X, sigma=self.Sigma
        ).fit()
        se_coef = results.bse[dim]

        # Compute z value for alpha
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        # Compute confidence interval
        return ConfidenceInterval(
            lower=self.point_estimate(dim) - z_alpha * se_coef,
            upper=self.point_estimate(dim) + z_alpha * se_coef,
        )

    def sd_estimate(self, dim: int) -> float:
        """
        Compute the standard deviation estimate for the estimator.
        """
        return (
            sm.GLS(self.training_data.y[:, 0], self.training_data.X, sigma=self.Sigma)
            .fit()
            .bse[dim]
        )


class GPBCI(Estimator):
    """
    Gaussian Process Bayesian Credible Intervals (GP BCIs) for spatial data.
    """

    def __init__(
        self,
        training_data: Dataset,
        test_data: Dataset,
        kernel: gpflow.kernels.Kernel,
        nugget_variance: float,
        max_likelihood: bool = True,
    ):
        super().__init__(training_data, test_data)
        self.kernel, self.noise_var, self.prior_variance = self.init_params(
            kernel, nugget_variance, max_likelihood
        )

    @staticmethod
    def name() -> str:
        return "GP BCIs"

    def init_params(self, kernel, nugget_variance, max_likelihood):
        """
        Initialize the covariance matrix Sigma.
        """
        # Compute the covariance matrix
        # Stack spatial locations and covariates, we will define the kernel to operate on spatial locations
        # And the mean function to act on covariates
        spatial_dim = self.training_data.S.shape[1]
        all_vars = np.concatenate((self.training_data.S, self.training_data.X), axis=1)
        # Define the kernel,
        linear_kernel = gpflow.kernels.Linear(
            active_dims=np.arange(
                spatial_dim, spatial_dim + self.training_data.X.shape[1]
            )
        )
        kernel = kernel + linear_kernel
        # Define the model
        model = gpflow.models.GPR(
            data=(all_vars, self.training_data.y),
            kernel=kernel,
            noise_variance=nugget_variance,
        )
        if max_likelihood:
            # Optimize the model
            opt = gpflow.optimizers.Scipy()
            opt.minimize(
                model.training_loss,
                model.trainable_variables,
                options=dict(maxiter=100),
            )

        return (
            model.kernel.kernels[0],
            model.likelihood.variance,
            model.kernel.kernels[1].variance,
        )

    def point_estimate(self, dim: int) -> float:
        """
        Compute the point estimate for the estimator.
        """
        all_covs = np.concatenate((self.training_data.S, self.training_data.X), axis=1)
        Sigma = self.kernel(all_covs, full_cov=True) + self.noise_var * np.eye(self.N)
        L = np.linalg.cholesky(Sigma)
        SigmaInvY = scipy.linalg.cho_solve((L, True), self.training_data.y)
        XTSigmaInvY = self.training_data.X.T @ SigmaInvY
        post_precision = self.posterior_precision()
        return np.linalg.solve(post_precision, XTSigmaInvY)[
            dim
        ]  # could cholesky, but this is probably fine.

    def posterior_precision(self) -> float:
        """
        Compute the posterior variance for the estimator.
        """
        all_covs = np.concatenate((self.training_data.S, self.training_data.X), axis=1)
        Sigma = self.kernel(all_covs, full_cov=True) + self.noise_var * np.eye(self.N)
        L = np.linalg.cholesky(Sigma)
        LinvX = scipy.linalg.solve_triangular(L, self.training_data.X, lower=True)
        XSigmaInvX = LinvX.T @ LinvX
        return XSigmaInvX + 1.0 / self.prior_variance * np.eye(self.P)

    def posterior_variance(self, dim: int) -> float:
        """
        Compute the posterior variance for the estimator coordinate.
        """
        ep = np.zeros((self.training_data.X.shape[1]))
        ep[dim] = 1
        posterior_precision = self.posterior_precision()
        L = np.linalg.cholesky(posterior_precision)
        LinvEp = scipy.linalg.solve_triangular(L, ep, lower=True)
        return np.sum(np.square(LinvEp))

    def confidence_interval(self, dim: int, alpha: float) -> ConfidenceInterval:
        """
        Compute the confidence interval for the OLS estimator.
        """
        mean = self.point_estimate(dim)
        # Compute the posterior variance
        std = np.sqrt(self.posterior_variance(dim))
        # Compute z value for alpha
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        # Compute confidence interval
        return ConfidenceInterval(
            lower=mean - z_alpha * std,
            upper=mean + z_alpha * std,
        )

    def sd_estimate(self, dim: int) -> float:
        """
        Compute the standard deviation estimate for the estimator.
        """
        return float(np.sqrt(self.posterior_variance(dim)))
