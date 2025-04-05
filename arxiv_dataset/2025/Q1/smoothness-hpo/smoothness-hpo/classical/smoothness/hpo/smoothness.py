import time

from functools import singledispatch
from typing import Callable, Tuple, Union

from smoothness.hpo.base import BaseHPO
from smoothness.hpo.util import get_learner, get_many_random_hyperparams

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


@singledispatch
def get_smoothness(learner, x_train, y_train):
    raise NotImplementedError


@get_smoothness.register(GaussianNB)
def _(learner: GaussianNB, x_train: np.array, y_train: np.array) -> float:
    # This function computes the gradient of the Hessian of the negative log-likelihood
    # of the Gaussian model.
    # The function returns the negative of the Frobenius norm of the gradient of the
    # Hessian of the negative log-likelihood of the Gaussian model.

    if len(y_train.shape) == 1:
        y = y_train.copy()
    else:
        y = np.argmax(y_train, axis=1)

    # First, we need to compute mu
    mu0 = np.mean(x_train[y == 0], axis=0)
    mu1 = np.mean(x_train[y == 1], axis=0)

    # Compute mu
    mu = np.empty_like(x_train)
    mu[y == 0] = mu0
    mu[y == 1] = mu1

    # Compute x_minus_mu
    x_minus_mu = (x_train - mu).T

    # Compute sigma
    sigma = np.matmul(x_minus_mu, x_minus_mu.T) / x_train.shape[0]

    if np.linalg.det(sigma) == 0:
        return 0
    sigma_inv = np.linalg.inv(sigma)

    # Compute identity tensor
    n = x_train.shape[1]
    I = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            I[i, j, i, j] = 1

    # Compute the first gradient G
    G = -sigma_inv @ (x_minus_mu @ x_minus_mu.T + 0.5 * sigma) @ sigma_inv

    # Compute the second gradient
    H = sigma_inv @ I @ G - 0.5 * sigma_inv @ I @ sigma_inv + G @ I @ sigma_inv

    # Return -norm of H since the minimum is better
    # We can't use norm(-H) since we're using the Frobenius norm
    return -np.linalg.norm(H)


@get_smoothness.register(LogisticRegression)
def _(learner: LogisticRegression, x_train: np.array, y_train: np.array) -> float:
    # Check number of classes
    if len(y_train.shape) > 1:
        y = np.argmax(y_train, axis=1)
    else:
        y = y_train.copy()

    learner.fit(x_train, y)

    # Get number of classes
    k = len(np.unique(y))

    # Compute smoothness
    return (k - 1) / (k * x_train.shape[0]) * np.linalg.norm(x_train) / np.linalg.norm(learner.coef_)


class SmoothnessHPO(BaseHPO):
    def __init__(self, hpo_space: dict, learner: str, query_fn: Callable[[dict], Union[tuple, float]],
                 data_fn: Callable[[dict], Tuple[np.array, np.array, np.array, np.array]]):
        super().__init__(hpo_space, learner, query_fn)
        self.data_fn = data_fn

    def run(self, n_runs: int, n_iters: int) -> Tuple[list, float]:
        """
        Runs random HPO.
        :param n_runs: Number of runs to perform.
        :param n_iters: Number of iterations to explore.
        :return:
        """
        start = time.time()

        scores = []
        keep_configs = 5
        for _ in range(n_runs):
            configs = get_many_random_hyperparams(self.hpo_space, n_iters)

            smoothness = []
            for config in configs:
                x_train, _, y_train, _ = self.data_fn(config)
                smoothness.append(get_smoothness(get_learner(self.learner, config), x_train, y_train))

            best_betas, best_configs = zip(*sorted(zip(smoothness, configs, strict=False), reverse=True, key=lambda x: x[0]), strict=False)
            best_configs = list(best_configs[:keep_configs])

            best_score = (-1,)

            for beta, config in zip(best_betas, best_configs, strict=False):
                score = self.query_fn(config)
                print('Beta:', beta, ' | Score:', score)

                if isinstance(score, float):
                    score = (score,)

                if score > best_score:
                    best_score = score

            scores.append(best_score)

        end = time.time()
        return scores, end - start
