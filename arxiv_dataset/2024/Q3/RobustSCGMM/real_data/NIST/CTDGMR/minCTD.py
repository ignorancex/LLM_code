import os
import sys
import copy
import time
import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from .distance import GMM_CTD
from .greedy import *
from .utils import *
from .barycenter import barycenter
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def robustmedian(
    subset_means,
    subset_covs,
    subset_weights,
    ground_distance="L2",
    coverage_ratio=0.5,
):
    n_split = len(subset_means)
    pairwisedist = np.zeros((n_split, n_split))
    for i in range(n_split):
        for j in range(n_split):
            if "CTD" in ground_distance:
                pairwisedist[i, j] = GMM_CTD(
                    [subset_means[i], subset_means[j]],
                    [subset_covs[i], subset_covs[j]],
                    [subset_weights[i], subset_weights[j]],
                    ground_distance=ground_distance.split("-")[1],
                    matrix=True)
            else:
                pairwisedist[i, j] = GMM_L2(
                    [subset_means[i], subset_means[j]],
                    [subset_covs[i], subset_covs[j]],
                    [subset_weights[i], subset_weights[j]],
                )
    # print(pairwisedist)
    which_GMM = np.argmin(np.quantile(pairwisedist, q=coverage_ratio, axis=1))

    output = [which_GMM, pairwisedist]

    return output


"""
Minimum composite transportation divergence (CTD) for GMR

Created by Qiong Zhang
"""


def entropy(log_ot_plan):
    """
    The entropy of a coupling matrix
    """

    return 1 - np.sum(np.exp(log_ot_plan) * log_ot_plan)


class GMR_CTD:
    """Find a GMM with n_components that is closest
    to a GMM parameterized by means, covs, weights in
    the composite transportation distance sense.

    Parameters
    ----------
    reg: strength of entropic regularization

    Returns
    -------
    weights and support points of reduced GMM.
    """

    def __init__(
        self,
        means,
        covs,
        weights,
        n,
        n_pseudo=100,
        init_method="kmeans",
        tol=1e-5,
        max_iter=100,
        ground_distance="KL",
        reg=0,
        means_init=None,
        covs_init=None,
        weights_init=None,
        random_state=0,
    ):
        self.means = means
        self.covs = covs
        self.weights = weights
        self.tol = tol
        self.max_iter = max_iter
        self.origin_n = self.weights.shape[0]
        self.new_n = n
        self.n_pseudo = n_pseudo
        self.random_state = random_state
        self.ground_distance = ground_distance
        self.converged_ = False
        if reg >= 0:
            self.reg = reg
        else:
            raise ValueError("The regularization term should be non-negative.")
        self.init_method = init_method
        self.means_init = copy.deepcopy(means_init)
        self.covs_init = copy.deepcopy(covs_init)
        self.weights_init = copy.deepcopy(weights_init)
        self.time_ = []

    def _initialize_parameter(self):
        """Initializatin of the clustering barycenter"""
        if self.init_method == "kmeans":
            total_sample_size = 1000
            X = rmixGaussian(
                self.means,
                self.covs,
                self.weights,
                total_sample_size,
                self.random_state,
            )[0]
            gm = GaussianMixture(n_components=self.new_n,
                                 random_state=self.random_state,
                                 tol=1e-6).fit(X)
            self.reduced_means = gm.means_
            self.reduced_covs = gm.covariances_
            self.reduced_weights = gm.weights_

        elif self.init_method == "user":
            self.reduced_means = self.means_init
            self.reduced_covs = self.covs_init
            self.reduced_weights = self.weights_init
        else:
            self.reduced_means, self.reduced_covs, self.reduced_weights = GMR_greedy(
                self.means, self.covs, self.weights, self.new_n,
                self.init_method)
        self.cost_matrix = GMM_CTD(
            means=[self.means, self.reduced_means],
            covs=[self.covs, self.reduced_covs],
            weights=[self.weights, self.reduced_weights],
            ground_distance=self.ground_distance,
            matrix=True,
            N=self.n_pseudo,
        )

    def _obj(self):
        if self.reg == 0:
            return np.sum(self.cost_matrix * self.ot_plan)
        elif self.reg > 0:
            return np.sum(self.cost_matrix *
                          self.ot_plan) - self.reg * entropy(self.log_ot_plan)

    def _weight_update(self):
        if self.reg == 0:
            self.clustering_matrix = (self.cost_matrix.T == np.min(
                self.cost_matrix, 1)).T
            self.ot_plan = self.clustering_matrix * (
                self.weights / self.clustering_matrix.sum(1)).reshape((-1, 1))
            # if there are ties, then the weights are equally splitted into
            # different groups
            self.reduced_weights = self.ot_plan.sum(axis=0)
        elif self.reg > 0:
            lognum = -self.cost_matrix / self.reg
            logtemp = (lognum.T - logsumexp(lognum, axis=1)).T
            self.log_ot_plan = (logtemp.T + np.log(self.weights)).T
            self.ot_plan = np.exp(self.log_ot_plan)
            self.reduced_weights = self.ot_plan.sum(axis=0)
        return self._obj()

    def _support_update(self):
        for i in range(self.new_n):
            self.reduced_means[i], self.reduced_covs[i] = barycenter(
                self.means,
                self.covs,
                self.ot_plan[:, i],
                mean_init=self.reduced_means[i],
                cov_init=self.reduced_covs[i],
                ground_distance=self.ground_distance,
            )
        self.cost_matrix = GMM_CTD(
            [self.means, self.reduced_means],
            [self.covs, self.reduced_covs],
            [self.weights, self.reduced_weights],
            ground_distance=self.ground_distance,
            matrix=True,
            N=self.n_pseudo,
        )
        return self._obj()

    def iterative(self):
        # print(np.squeeze(self.means))
        self._initialize_parameter()
        obj = np.Inf
        for n_iter in range(1, self.max_iter + 1):
            proc_time = time.time()
            obj_current = self._weight_update()
            # remove the empty cluster centers
            index = np.where(self.ot_plan.sum(axis=0) != 0)
            self.new_n = index[0].shape[0]
            self.ot_plan = self.ot_plan.T[index].T
            self.reduced_means = self.reduced_means[index[0]]
            self.reduced_covs = self.reduced_covs[index[0]]
            self.reduced_weights = self.reduced_weights[index[0]]

            if n_iter > 1:
                change = (obj - obj_current) / np.max([obj, obj_current, 1])
            else:
                change = obj - obj_current
            if change < self.tol:
                self.converged_ = True
                self.obj = obj
                self.n_iter_ = n_iter
                # print(np.squeeze(self.means))
                break

            if change < 0.0:
                raise ValueError(
                    "Weight update: The objective function is increasing!")
            obj = obj_current
            obj_current = self._support_update()
            change = (obj - obj_current) / np.max([obj, obj_current, 1])
            self.time_.append(time.time() - proc_time)
            if change < self.tol:
                self.converged_ = True
                self.obj = obj
                self.n_iter_ = n_iter
                break
            if change < 0.0:
                raise ValueError(
                    "Support update: The objective function is increasing!")
            obj = obj_current

        if not self.converged_:
            print("Algorithm did not converge. "
                  "Try different init parameters, "
                  "or increase max_iter, tol ")


class GMR_PCTD:
    """Find a GMM with n_components that is closest
    to a GMM parameterized by means, covs, weights in
    the partial composite transportation distance sense.

    Parameters
    ----------
    means: array-like (m, d)
    covs: array-like (m, d, d)
    weights: array-like (m, )
    alpha: parameter in PCTD [0,1)

    Returns
    -------
    weights and component parameters of reduced GMM.
    """

    def __init__(
        self,
        means,
        covs,
        weights,
        n,
        init_method="kmeans",
        tol=1e-5,
        max_iter=100,
        ground_distance="KL",
        alpha=0,
        means_init=None,
        covs_init=None,
        weights_init=None,
        random_state=0,
    ):
        self.means = means
        self.covs = covs
        self.weights = weights
        self.tol = tol
        self.max_iter = max_iter
        self.origin_n = self.weights.shape[0]
        self.new_n = n
        self.random_state = random_state
        self.ground_distance = ground_distance
        self.converged_ = False
        if alpha >= 0 and alpha < 1.0:
            self.alpha = alpha
        else:
            raise ValueError("The parameter in PCTD should in [0, 1).")
        self.init_method = init_method
        self.means_init = copy.deepcopy(means_init)
        self.covs_init = copy.deepcopy(covs_init)
        self.weights_init = copy.deepcopy(weights_init)
        self.time_ = []

    def _initialize_parameter(self):
        """Initializatin of the clustering barycenter"""
        if self.init_method == "kmeans":
            total_sample_size = 1000
            X = rmixGaussian(
                self.means,
                self.covs,
                self.weights,
                total_sample_size,
                self.random_state,
            )[0]
            gm = GaussianMixture(n_components=self.new_n,
                                 random_state=self.random_state,
                                 tol=1e-6).fit(X)
            self.reduced_means = gm.means_
            self.reduced_covs = gm.covariances_
            self.reduced_weights = gm.weights_

        elif self.init_method == "user":
            self.reduced_means = copy.deepcopy(self.means_init)
            self.reduced_covs = copy.deepcopy(self.covs_init)
            self.reduced_weights = copy.deepcopy(self.weights_init)
        else:
            self.reduced_means, self.reduced_covs, self.reduced_weights = GMR_greedy(
                self.means, self.covs, self.weights, self.new_n,
                self.init_method)
        # self.cost_matrix = GMM_CTD(
        #     means=[self.means, self.reduced_means],
        #     covs=[self.covs, self.reduced_covs],
        #     weights=[self.weights, self.reduced_weights],
        #     ground_distance=self.ground_distance,
        #     matrix=True,
        # )

    def _obj(self):
        return np.sum(self.track * self.min_dist)

    def _weight_update(self):
        # for i in range(self.reduced_covs.shape[0]):
        #     print(np.linalg.eigvals(self.reduced_covs[i]).min())

        self.cost_matrix = GMM_CTD(
            means=[self.means, self.reduced_means],
            covs=[self.covs, self.reduced_covs],
            weights=[self.weights, self.reduced_weights],
            ground_distance=self.ground_distance,
            matrix=True,
        )
        self.min_dist = np.min(self.cost_matrix, axis=1)
        self.sorted_index = np.argsort(self.min_dist)

        # sort the weight
        self.sorted_weights = self.weights[self.sorted_index]

        # find tau_alpha
        self.hs = np.where(
            np.cumsum(self.sorted_weights) >= 1 - self.alpha -
            1e-16)[0][0]  # - np.finfo(float).eps
        # self.hs = np.where(np.cumsum(self.sorted_weights) >= 1 - self.alpha)[0][
        #     0
        # ]

        # print(self.hs)
        self.track = copy.deepcopy(self.weights)

        # print(np.cumsum(self.sorted_weights)[self.hs], 1-self.alpha)

        # if np.cumsum(self.sorted_weights)[self.hs] == 1-self.alpha:
        #     self.track[self.sorted_index[(self.hs + 1):]] = 0
        # else:
        #     self.track[self.sorted_index[self.hs]] = (
        #         1 - self.alpha - np.cumsum(self.sorted_weights)[self.hs-1]
        #     )
        #     if self.hs + 1 <= self.new_n - 1:
        #         self.track[self.sorted_index[(self.hs + 1):]] = 0
        self.track[self.sorted_index[(self.hs + 1):]] = 0
        self.track[self.sorted_index[self.hs]] = (
            1 - self.alpha - np.cumsum(self.sorted_weights)[self.hs - 1])

        # print(self.track.sum())

        # the components that are trimmed
        self.trimmed_label = np.zeros(self.means.shape[0])
        self.trimmed_label[self.sorted_index[(self.hs + 1):]] = 1
        # shape of clustering matrix is (n*k, k)
        self.clustering_matrix = (self.cost_matrix.T == self.min_dist).T
        self.ot_plan = self.clustering_matrix * self.track.reshape((-1, 1))

        self.reduced_weights = np.sum(self.ot_plan, axis=0) / (1 - self.alpha)

        # print(np.sum(
        #     self.ot_plan, axis=0).sum(), self.alpha, self.reduced_weights.sum())
        return self._obj()

    def _support_update(self):
        for i in range(self.new_n):
            self.reduced_means[i], self.reduced_covs[i] = barycenter(
                self.means,
                self.covs,
                self.ot_plan[:, i],
                mean_init=self.reduced_means[i],
                cov_init=self.reduced_covs[i],
                ground_distance=self.ground_distance,
            )
            # print(self.ot_plan[:, i])

    def iterative(self):
        self._initialize_parameter()
        obj = np.Inf
        for n_iter in range(1, self.max_iter + 1):
            # print('iteration', n_iter)
            proc_time = time.time()
            obj_current = self._weight_update()
            self._support_update()
            self.time_.append(time.time() - proc_time)

            # remove the empty cluster centers
            index = np.where(self.ot_plan.sum(axis=0) != 0)
            self.new_n = index[0].shape[0]
            self.ot_plan = self.ot_plan.T[index].T
            self.reduced_means = self.reduced_means[index[0]]
            self.reduced_covs = self.reduced_covs[index[0]]
            self.reduced_weights = self.reduced_weights[index[0]]

            if n_iter > 1:
                change = (obj - obj_current) / np.max([obj, obj_current, 1])
            else:
                change = obj - obj_current

            if change < self.tol:
                self.converged_ = True
                self.obj = obj
                self.n_iter_ = n_iter
                break
            if change < 0.0:
                raise ValueError(
                    "Weight update: The objective function is increasing!")
            obj = obj_current

        if not self.converged_:
            print("Algorithm did not converge. "
                  "Try different init parameters, "
                  "or increase max_iter, tol ")
