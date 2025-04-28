from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from collections import namedtuple
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import brentq
import scipy.stats as stats
from ot import emd2
from abc import abstractmethod
from .estimators import Estimator, Dataset, ConfidenceInterval
from ..estimate_variance import estimate_minimum_variance, fast_estimate_variance
from sklearn.metrics.pairwise import haversine_distances


class LipschitzDrivenEstimator(Estimator):
    def __init__(
        self,
        training_data: Dataset,
        test_data: Dataset,
        lipschitz_bound: float,
        noise_std: Optional[float] = None,
        fast_noise: Optional[bool] = False,
        data_on_sphere: Optional[bool] = False,
    ):
        super().__init__(training_data, test_data)
        self.lipschitz_bound = lipschitz_bound
        self.data_on_sphere = data_on_sphere
        self.noise_std = (
            noise_std if noise_std is not None else np.sqrt(self.estimate_noise_variance(fast_noise=fast_noise))
        )
        self._Psi = self._build_Psi()

    @staticmethod
    def name() -> str:
        return f"Ours"

    def point_estimate(self, dim: int) -> float:
        """
        Compute the point estimate for the estimator.
        """
        return np.sum(self.v(dim) * (self.Psi @ self.training_data.y))

    def bias_bdd(self, dim: int) -> float:
        """
        Compute an upper bound on the bias of the estimator. This is done by using the Lipschitz assumption. We have
        that the bias is bounded by the Lipschitz constant times a Wasserstein distance.
        sup_f |sum_m v_m f(S*_m) - sum_n w_n f(S*_n)|.
        Because coefficients may be positive or negative (and might not sum to 1), we sort them,
        divide into positive and negative coeffs, normalize,
        then solve the corresponding Wasserstein distance calculation via linear programming or some other method.
        Anything that returns an upper bound on the Wasserstein distance is ok with us!
        """

        # Get the weights
        v = self.v(dim)
        w = self.w(dim)
        # Coefficients of the linear program. Take a negative because we want to maximize the objective.
        all_coeffs = np.concatenate([v, -w])
        # Weights should sum to 1, or else shifting leads to unbounded loss.
        if not np.isclose(np.sum(all_coeffs), 0):
            return np.inf
        # get indices of positive coefficients
        pos_indices = np.where(all_coeffs > 0)[0]
        # get indices of negative coefficients
        neg_indices = np.where(all_coeffs <= 0)[0]
        # Get the corresponding covariates
        all_covariates = np.concatenate([self.test_data.S, self.training_data.S])
        pos_covariates = all_covariates[pos_indices]
        neg_covariates = all_covariates[neg_indices]
        pos_coeffs = all_coeffs[pos_indices]
        neg_coeffs = all_coeffs[neg_indices]
        normalization = np.sum(pos_coeffs[:, 0])
        # If all the weights are 0, return 0.
        if normalization <= 1e-12:
            return 0.0
        pos_coeffs /= normalization
        neg_coeffs /= normalization

        if self.data_on_sphere:
            # Use haversine distance for spherical data, multiply by Earth's radius to get km
            dist_matrix = haversine_distances(pos_covariates, neg_covariates) * 6371
        else:
            # Compute the euclidean distance
            dist_matrix = np.linalg.norm(
                pos_covariates[:, None] - neg_covariates, axis=-1
            )
        # Compute wasserstein distance
        wass = emd2(pos_coeffs[:, 0], -neg_coeffs[:, 0], dist_matrix)

        return self.lipschitz_bound * wass * normalization

    def estimate_noise_variance(self, fast_noise: bool = False) -> float:
        """
        Estimate the noise variance of the model. This is either done via LOOCV with nearest neighbors
        if fast_noise is True, or via ERM over the class of Lipschitz functions if fast_noise is False.
        """
        estimate_variance_fn = (
            fast_estimate_variance if fast_noise else estimate_minimum_variance
        )
        return estimate_variance_fn(
            self.training_data.S,
            self.training_data.y,
            self.lipschitz_bound,
            self.data_on_sphere,
        )

    def confidence_interval(self, dim: int, alpha: float = 0.05) -> ConfidenceInterval:
        """
        We compute confidence intervals by using the bias and the randomness bound.
        We then perform root finding to find the correct Delta that satisfies the confidence level.
        See the paper for details.
        """
        point_estimate = self.point_estimate(dim)
        bias = self.bias_bdd(dim)
        randomness_std = self.randomness_std_bdd(dim)
        if randomness_std == 0:
            return ConfidenceInterval(
                lower=point_estimate - bias, upper=point_estimate + bias
            )

        def fn(delta):
            right_tail = stats.norm.cdf(delta / randomness_std)
            left_tail = stats.norm.cdf(-(bias + delta) / randomness_std)
            return right_tail - left_tail - (1 - alpha)

        delta_upper = stats.norm.ppf(1 - alpha / 2) * randomness_std
        delta_lower = stats.norm.ppf(1 - alpha) * randomness_std
        randomness_tails = brentq(fn, delta_lower, delta_upper)

        return ConfidenceInterval(
            lower=point_estimate - bias - randomness_tails,
            upper=point_estimate + bias + randomness_tails,
        )

    def v(self, dim: int) -> ArrayLike:
        """
        Compute the v vector for the linear estimator, v= ep^T (X*^T X*)^-1 X*^T.
        """
        # Moore-Penrose pseudoinverse is (X*^T X*)^-1 X*^T. Pickout the dim-th row.
        return np.linalg.pinv(self.test_data.X)[dim][:, None]

    def randomness_std_bdd(self, dim: int) -> float:
        """
        Compute the randomness bound for the confidence interval.
        """
        return self.noise_std * np.linalg.norm(self.w(dim), ord=2)

    @abstractmethod
    def _build_Psi(self) -> ArrayLike:
        """
        Compute the coupling matrix Psi for the estimator.
        """
        raise NotImplementedError

    @property
    def Psi(self) -> ArrayLike:
        """
        Compute the coupling matrix Psi for the estimator.
        """
        return self._Psi

    def w(self, dim: int) -> ArrayLike:
        """
        Compute the weights w for the linear estimator.
        """
        return self.Psi.T @ self.v(dim)

    def sd_estimate(self, dim: int) -> float:
        """
        Compute the standard deviation estimate for the estimator.
        """
        return self.randomness_std_bdd(dim)


class NNLipschitzDrivenEstimator(LipschitzDrivenEstimator):

    def __init__(
        self,
        training_data: Dataset,
        test_data: Dataset,
        lipschitz_bound: float,
        noise_std: Optional[float] = None,
        fast_noise: Optional[bool] = False,
        data_on_sphere: Optional[bool] = False,
        num_neighbors: Optional[int] = 1,
    ):
        self.num_neighbors = num_neighbors
        super().__init__(
            training_data,
            test_data,
            lipschitz_bound,
            noise_std,
            fast_noise,
            data_on_sphere,
        )

    def _build_Psi(self) -> ArrayLike:
        """
        Compute the coupling matrix Psi for the nearest neighbor coupling.
        """
        if self.data_on_sphere:
            # Use haversine distance for spherical data
            nn = NearestNeighbors(n_neighbors=self.num_neighbors, metric="haversine")
        else:
            nn = NearestNeighbors(n_neighbors=self.num_neighbors)

        nn.fit(self.training_data.S)
        # Get the indices of the nearest neighbors
        _, indices = nn.kneighbors(self.test_data.S)
        # Set the coupling matrix by scattering 1/num_neighbors to the correct indices
        Psi = np.zeros((self.test_data.S.shape[0], self.training_data.S.shape[0]))
        Psi[np.arange(self.test_data.S.shape[0])[:, None], indices] = (
            1 / self.num_neighbors
        )
        return Psi
