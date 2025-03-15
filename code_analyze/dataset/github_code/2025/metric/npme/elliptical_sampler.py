from typing import Callable
import numpy as np
from tqdm import tqdm

class EllipticalSliceSampler:
    def __init__(self,
                 prior_cov: np.ndarray,
                 loglik: Callable):
        """
        Initializes the Elliptical Slice Sampler.
        
        Args:
        - prior_cov (np.ndarray): Prior covariance matrix.
        - loglik (Callable): Log-likelihood function.
        """
        self.prior_cov = prior_cov
        self.loglik = loglik

        self._n = prior_cov.shape[0]  # Dimensionality of the space
        self._chol = np.linalg.cholesky(prior_cov)  # Cache Cholesky decomposition

        # Initialize state by sampling from prior
        self._state_f = self._chol @ np.random.randn(self._n)

    def _indiv_sample(self):
        """
        Main algorithm for generating an individual sample using Elliptical Slice Sampling.
        """
        f = self._state_f  # Previous state
        nu = self._chol @ np.random.randn(self._n)  # Sample from prior for the ellipse
        log_y = self.loglik(f) + np.log(np.random.uniform())  # Log-likelihood threshold

        theta = np.random.uniform(0., 2 * np.pi)  # Initial proposal angle
        theta_min, theta_max = theta - 2 * np.pi, theta  # Define bracketing interval

        # Main loop: Accept sample if it meets log-likelihood threshold; otherwise, shrink the bracket.
        while True:
            # YOUR CODE HERE (~10 lines)
            # 1. Generate a new sample point based on the current angle.
            # 2. Check if the proposed point meets the acceptance criterion.            
            # 3. If not accepted, adjust the bracket and select a new angle.
            f_new = f * np.cos(theta) + nu * np.sin(theta) 
            if self.loglik(f_new) > log_y:
                self._state_f = f_new
                return
            if theta < 0:
                theta_min = theta
            else:
                theta_max = theta
            theta = np.random.uniform(theta_min, theta_max)
            # END OF YOUR CODE

    def sample(self,
               n_samples: int,
               n_burn: int = 500) -> np.ndarray:
        """
        Generates samples using Elliptical Slice Sampling.

        Args:
        - n_samples (int): Total number of samples to return.
        - n_burn (int): Number of initial samples to discard (burn-in).

        Returns:
        - np.ndarray: Array of samples after burn-in.
        """
        samples = []
        for i in tqdm(range(n_samples), desc="Sampling"):
            self._indiv_sample()
            if i > n_burn:
                samples.append(self._state_f.copy())  # Store sample post burn-in

        return np.stack(samples)
