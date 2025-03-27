from abc import ABCMeta, abstractmethod

import jax.numpy as jnp
import numpy as np
import torch
from jax import grad, vmap
from torchtyping import TensorType
from typing import Tuple

class ScoringRule(metaclass=ABCMeta):
    """This is the abstract class for the ScoringRule. I removed the summary statistics calculation which is usually
    done in ABCpy, but it is redundant here."""

    def __init__(self, weight=1):
        """Needs to be called by each sub-class to correctly initialize the statistics_calc"""
        self.weight = weight # this is the weight used to multiply the scoring rule for the loglikelihood computation

    def loglikelihood(self, y_obs, y_sim, use_torch=False):
        """Alias the score method to a loglikelihood method """
        return - self.weight * self.score(y_obs, y_sim, use_torch=use_torch)

    def score(self, observations, simulations, use_torch=False):
        """
        Notice: here the score is assumed to be a "penalty"; we use therefore the sign notation of Dawid, not the one
        in Gneiting and Raftery (2007).
        To be overwritten by any sub-class. Here, `observations` and `simulations` are of length respectively `n_obs` and `n_sim`. Then,
        for each fixed observation the `n_sim` simulations are used to estimate the scoring rule. Subsequently, the
        values are summed over each of the `n_obs` observations.

        Parameters
        ----------
        observations: numpy array or torch tensor or list
            Contains `n_obs` data points.
        simulations: numpy array or torch tensor or list
            Contains `n_sim` data points.

        Returns
        -------
        float
            The score between the simulations and the observations.

        Notes
        -----
        When running an ABC algorithm, the observed dataset is always passed first to the distance. Therefore, you can
        save the statistics of the observed dataset inside this object, in order to not repeat computations.
        """

        if use_torch:
            return self._estimate_score_torch_fast(simulations=simulations, observations=observations)
            
        observations = np.array(observations)
        simulations = np.array(simulations)

        return self._estimate_score_numpy(observations, simulations)

    @abstractmethod
    def _estimate_score_numpy(self, s_observations, s_simulations):
        """
        This method needs to be implemented by each sub-class. It should return the score for the given data set.

        Parameters
        ----------
        s_observations: numpy array
            The summary statistics of the observed data set. Shape is (n_obs, n_summary_stat).
        s_simulations: numpy array
            The summary statistics of the simulated data set. Shape is (n_sim, n_summary_stat).
        """
        raise NotImplementedError

    @abstractmethod
    def _estimate_score_torch(self, simulations: TensorType["n_simulations", "data_size"],
                              observations: TensorType["n_observations", "data_size"]) -> TensorType[float]:
        """
        Add docstring
        """

        raise NotImplementedError

    @abstractmethod
    def score_max(self):
        """To be overwritten by any sub-class"""
        raise NotImplementedError


class EnergyScore(ScoringRule):
    def __init__(self, beta=1, use_jax=False, weight=1):
        """ Estimates the EnergyScore. Here, I assume the observations and simulations are lists of
        length respectively n_obs and n_sim. Then, for each fixed observation the n_sim simulations are used to estimate the
        scoring rule. Subsequently, the values are summed over each of the n_obs observations.

        Note this scoring rule is connected to the energy distance between probability distributions.
        Parameters
        ----------
        beta : int, optional.
            Power used to define the energy score. Default is 1.
        use_jax : bool, optional.
            Whether to use JAX for the computation; in that case, you can compute unbiased gradient estimate
            of the score with respect to parameters. Default is False.
        """

        self.beta = beta
        self.beta_over_2 = 0.5 * beta
        self.use_jax = use_jax

        if use_jax:
            # define the gradient function with jax:
            self._grad_estimate_score = grad(self._estimate_score_numpy, argnums=1)
            self.np = jnp
        else:
            self.np = np

        super(EnergyScore, self).__init__(weight)

    def score_gradient(self, observations, simulations, simulations_gradients):
        """
        Computes gradient of the unbiased estimate of the score with respect to the parameters.

        Parameters
        ----------
        observations: numpy array or torch tensor
            Contains n1 data points.
        simulations: numpy array or torch tensor
            Contains n2 data points.
        simulations_gradients: numpy array or torch tensor
            Contains n2 data points, each of which is the gradient of the simulations with respect to the
            parameters (and is therefore of shape (simulation_dim, n_params)).

        Returns
        -------
        numpy.ndarray
            The gradient of the score with respect to the parameters.

        Notes
        -----
        When running an ABC algorithm, the observed dataset is always passed first to the distance. Therefore, you can
        save the statistics of the observed dataset inside this object, in order to not repeat computations.
        """
        if not self.use_jax:
            raise RuntimeError("The gradient of the energy score is only available with jax.")

        if not len(simulations) == len(simulations_gradients):
            raise RuntimeError("The number of simulations and the number of gradients must be the same.")

        observations = np.array(observations)
        simulations = np.array(simulations)

        score_grad = self._grad_estimate_score(observations, simulations)
        # score grad contains the gradients of the score with respect to the simulation statistics; it is therefore of
        # shape (n_sim, simulation_dim)
        simulations_gradients = np.array(simulations_gradients)  # maybe this is not needed
        # simulations_gradients is of shape (n_sim, simulation_dim, n_params)

        if not simulations_gradients.shape[0:2] == score_grad.shape:
            raise RuntimeError("The shape of the score gradient must be the"
                               " same as the first two shapes of the simulations.")

        # then need to multiply the gradients for each simulation along the simulation_dim axis and then average
        # over n_dim; that leads to the gradient of the score with respect to the parameters:

        return np.einsum('ij,ijk->k', score_grad, simulations_gradients)

    def score_max(self):
        # As the statistics are positive, the max possible value is 1
        return np.inf

    def _estimate_score_numpy(self, s_observations, s_simulations):
        """
        We estimate this by building an empirical unbiased estimate of Eq. (2) in Ziel and Berk 2019

        Parameters
        ----------
        s_observations: numpy array
            The summary statistics of the observed data set. Shape is (n_obs, n_summary_stat).
        s_simulations: numpy array
            The summary statistics of the simulated data set. Shape is (n_sim, n_summary_stat).
        """
        n_obs = s_observations.shape[0]
        n_sim, p = s_simulations.shape
        diff_X_y = s_observations.reshape(n_obs, 1, -1) - s_simulations.reshape(1, n_sim, p)
        diff_X_y = self.np.einsum('ijk, ijk -> ij', diff_X_y, diff_X_y)

        diff_X_tildeX = s_simulations.reshape(1, n_sim, p) - s_simulations.reshape(n_sim, 1, p)

        # exclude diagonal elements which are zero:
        diff_X_tildeX = self.np.einsum('ijk, ijk -> ij', diff_X_tildeX, diff_X_tildeX)[~self.np.eye(n_sim, dtype=bool)]
        if self.beta_over_2 != 1:
            diff_X_y **= self.beta_over_2
            diff_X_tildeX **= self.beta_over_2

        return 2 * self.np.sum(self.np.mean(diff_X_y, axis=1)) - n_obs * self.np.sum(diff_X_tildeX) / (
                n_sim * (n_sim - 1))

    def _estimate_score_torch(self, simulations: TensorType["n_simulations", "data_size"],
                              observations: TensorType["n_observations", "data_size"]) -> TensorType[float]:
        return sum([self._estimate_score_torch_batch(simulations.unsqueeze(0), observations[i].unsqueeze(0)) for i in
                    range(observations.shape[0])])

    def _estimate_score_torch_fast(self, simulations, observations):
        beta_over_2 = self.beta / 2

        n_obs = observations.shape[0]
        n_sim, p = simulations.shape
        diff_X_y = observations.reshape(n_obs, 1, -1) - simulations.reshape(1, n_sim, p)
        diff_X_y = torch.einsum('ijk, ijk -> ij', diff_X_y, diff_X_y)

        diff_X_tildeX = simulations.reshape(1, n_sim, p) - simulations.reshape(n_sim, 1, p)

        # exclude diagonal elements which are zero:
        diff_X_tildeX = torch.einsum('ijk, ijk -> ij', diff_X_tildeX, diff_X_tildeX)[~torch.eye(n_sim, dtype=bool)]

        if beta_over_2 != 1:
            diff_X_y **= beta_over_2
            diff_X_tildeX **= beta_over_2

        return 2 * torch.sum(torch.mean(diff_X_y, axis=1)) - n_obs * torch.sum(diff_X_tildeX) / (n_sim * (n_sim - 1))

    def _estimate_score_torch_batch(self, simulations: TensorType["batch", "ensemble_size", "data_size"],
                                    observations: TensorType["batch", "data_size"]) -> TensorType[float]:
        """The previous implementation considered a set of simulations and a set of observations, and estimated the
        score separately for each observation with the provided simulations. Here instead we have a batch
        of (simulations, observation); then it corresponds to the one above when batch_size=1 and the observation size
        is =1. We want therefore an implementation which works parallely over batches."""

        batch_size, ensemble_size, data_size = simulations.shape

        # old version: the gradient computation when using this failed, when taking the power of diff_X_tildeX, due to
        # that matrix containing 0 entries; if self.beta_over_2 < 1, the gradient had a 0 term in the denominator, which
        # lead to nan values. The new version uses a torch command which computes the pairwise distances and does not
        # lead to nan gradients. It is also slightly faster.
        # diff_X_y = observations.reshape(batch_size, 1, data_size) - simulations
        # diff_X_y = torch.einsum('bep, bep -> be', diff_X_y, diff_X_y)
        #
        # diff_X_tildeX = simulations.reshape(batch_size, 1, ensemble_size, data_size) - (simulations.reshape(
        #     batch_size, ensemble_size, 1,
        #     data_size))  # idea could be adding an epsilon for numerical stability, but does not seem to work.
        # diff_X_tildeX = torch.einsum('befp, befp -> bef', diff_X_tildeX, diff_X_tildeX)
        #
        # if self.beta_over_2 != 1:
        #     diff_X_y = torch.pow(diff_X_y, self.beta_over_2)
        #     diff_X_tildeX = torch.pow(diff_X_tildeX, self.beta_over_2)

        # the following should have shape  ["batch", "ensemble_size", "data_size"], contains all differences of each
        # observations from its own simulationss
        diff_X_y = torch.cdist(observations.reshape(batch_size, 1, data_size), simulations, p=2)
        diff_X_y = torch.squeeze(diff_X_y, dim=1)

        # the following should have shape  ["batch", "ensemble_size", "ensemble_size", "data_size"], contains all
        # differences of each observations from each other observations for each batch element
        diff_X_tildeX = torch.cdist(simulations, simulations, p=2)

        if self.beta != 1:
            diff_X_tildeX = torch.pow(diff_X_tildeX, self.beta)
            diff_X_y = torch.pow(diff_X_y, self.beta)

        result = 2 * torch.sum(torch.mean(diff_X_y, dim=1)) - torch.sum(diff_X_tildeX) / (
                ensemble_size * (ensemble_size - 1))

        return result


class KernelScore(ScoringRule):

    def __init__(self, weight=1, kernel="gaussian", biased_estimator=False, use_jax=False, **kernel_kwargs):
        """
        Parameters
        ----------
        kernel : str or callable, optional
            Can be a string denoting the kernel, or a function. If a string, only gaussian is implemented for now; in
            that case, you can also provide an additional keyword parameter 'sigma' which is used as the sigma in the
            kernel. If a function is provided it should take two arguments; additionally, it needs to be written in
            jax if use_jax is True, otherwise gradient computation will not work.
        biased_estimator : bool, optional
            Whether to use the biased estimator or not. Default is False.
        use_jax : bool, optional.
            Whether to use JAX for the computation; in that case, you can compute unbiased gradient estimate
            of the score with respect to parameters. Default is False.
        **kernel_kwargs : dict, optional
            Additional keyword arguments for the kernel.
        """

        if not isinstance(kernel, str) and not callable(kernel):
            raise RuntimeError("'kernel' must be either a string or a function of two variables returning a scalar. "
                               "In that case, it must be written in JAX if use_jax is True.")

        super(KernelScore, self).__init__(weight)

        self.kernel_vectorized = False
        self.use_jax = use_jax

        if use_jax:
            # define the gradient function with jax:
            self._grad_estimate_score = grad(self._estimate_score_numpy, argnums=1)
            self.np = jnp
        else:
            self.np = np

        # set up the kernel
        if isinstance(kernel, str):
            if kernel == "gaussian":
                if 'sigma' in kernel_kwargs:
                    self.sigma = kernel_kwargs['sigma']
                else:
                    self.sigma = 1
                    
                self.kernel = self._def_gaussian_kernel(**kernel_kwargs)
                self.kernel_vectorized = True  # the gaussian kernel is vectorized
                # the following is the kernel for the torch setup:
                self.kernel_torch = self._def_gaussian_kernel_torch(**kernel_kwargs)
            else:
                raise NotImplementedError("The required kernel is not implemented.")
        else:  # if kernel is a callable already
            if use_jax:
                self.kernel = self._all_pairs(kernel)  # this makes it a vectorized function
                self.kernel_vectorized = True  # the gaussian kernel is vectorized
            else:
                self.kernel = self.kernel_torch = kernel

        self.biased_estimator = biased_estimator

    def score_gradient(self, observations, simulations, simulations_gradients):
        """
        Computes gradient of the unbiased estimate of the score with respect to the parameters.

        Parameters
        ----------
        observations: numpy array or torch tensor
            Contains n1 data points.
        simulations: numpy array or torch tensor
            Contains n2 data points.
        simulations_gradients: numpy array or torch tensor
            Contains n2 data points, each of which is the gradient of the simulations with respect to the
            parameters (and is therefore of shape (simulation_dim, n_params)).

        Returns
        -------
        numpy.ndarray
            The gradient of the score with respect to the parameters.

        Notes
        -----
        When running an ABC algorithm, the observed dataset is always passed first to the distance. Therefore, you can
        save the statistics of the observed dataset inside this object, in order to not repeat computations.
        """
        if not self.use_jax:
            raise RuntimeError("The gradient of the energy score is only available with jax.")

        if not len(simulations) == len(simulations_gradients):
            raise RuntimeError("The number of simulations and the number of gradients must be the same.")

        observations = np.array(observations)
        simulations = np.array(simulations)

        score_grad = self._grad_estimate_score(observations, simulations)
        # score grad contains the gradients of the score with respect to the simulation statistics; it is therefore of
        # shape (n_sim, simulation_dim)
        simulations_gradients = np.array(simulations_gradients)  # maybe this is not needed
        # simulations_gradients is of shape (n_sim, simulation_dim, n_params)

        if not simulations_gradients.shape[1] == score_grad.shape[1]:
            raise RuntimeError("The shape of the score gradient must be the"
                               " same as the first two shapes of the simulations.")

        # then need to multiply the gradients for each simulation along the simulation_dim axis and then average
        # over n_dim; that leads to the gradient of the score with respect to the parameters:

        return np.einsum('ij,ijk->k', score_grad, simulations_gradients)

    def score_max(self):
        """
        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """

        # As the statistics are positive, the max possible value is 1
        return np.inf

    def _estimate_score_numpy(self, s_observations, s_simulations):
        """
        Parameters
        ----------
        s_observations: numpy array
            The summary statistics of the observed data set. Shape is (n_obs, n_summary_stat).
        s_simulations: numpy array
            The summary statistics of the simulated data set. Shape is (n_sim, n_summary_stat).
        """
        # compute the Gram matrix
        K_sim_sim, K_obs_sim = self._compute_Gram_matrix(s_observations, s_simulations)

        # Estimate MMD
        if self.biased_estimator:
            return self._MMD_V_estimator(K_sim_sim, K_obs_sim)
        else:
            return self._MMD_unbiased(K_sim_sim, K_obs_sim)

    def _estimate_score_torch(self, simulations: TensorType["n_simulations", "data_size"],
                              observations: TensorType["n_observations", "data_size"]) -> TensorType[float]:
        return sum([self._estimate_score_torch_batch(simulations.unsqueeze(0), observations[i].unsqueeze(0)) for i in
                    range(observations.shape[0])])

    def _estimate_score_torch_fast(self, simulations, observations):
        if self.biased_estimator:
            raise NotImplementedError("Only unbiased estimator implemented!")

        def _gaussian_kernel_vectorized(X, Y, sigma):
            sigma_2 = 2 * sigma ** 2
            """Here X and Y have shape (n_samples_x, n_features) and (n_samples_y, n_features);
            this directly computes the kernel for all pairwise components"""
            XY = X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1)  # pairwise differences
            #Y = Y.to(torch.float)
            #XY = torch.cdist(X, Y)
            #return torch.exp(- torch.pow(XY, 2) / sigma_2)

            return torch.exp(- torch.einsum('xyi,xyi->xy', XY, XY) / sigma_2)


        def _MMD_unbiased(K_sim_sim, K_obs_sim):
            # Adapted from https://github.com/eugenium/MMD/blob/2fe67cbc7378f10f3b273cfd8d8bbd2135db5798/mmd.py
            # The estimate when distribution of x is not equal to y
            n_obs, n_sim = K_obs_sim.shape

            t_obs_sim = (2. / n_sim) * torch.sum(K_obs_sim)
            t_sim_sim = (1. / (n_sim * (n_sim - 1))) * torch.sum(K_sim_sim[~torch.eye(n_sim, dtype=bool)])

            return n_obs * t_sim_sim - t_obs_sim

        # compute the Gram matrix
        K_sim_sim, K_obs_sim = (_gaussian_kernel_vectorized(simulations, simulations, self.sigma),
                                _gaussian_kernel_vectorized(observations, simulations, self.sigma))

        return _MMD_unbiased(K_sim_sim, K_obs_sim)

    def _estimate_score_torch_batch(self, simulations: TensorType["batch", "ensemble_size", "data_size"],
                                    observations: TensorType["batch", "data_size"]) -> TensorType[float]:
        """The previous implementation considered a set of simulations and a set of observations, and estimated the
        score separately for each observation with the provided simulations. Here instead we have a batch
        of (simulations, observation); then it corresponds to the one above when batch_size=1 and the observation size
        is =1. We want therefore an implementation which works parallely over batches."""

        # compute the Gram matrix
        K_sim_sim, K_obs_sim = self._compute_Gram_matrix_torch(simulations, observations)

        # Estimate MMD
        if self.biased_estimator:
            result = self._MMD_V_estimator_torch(K_sim_sim, K_obs_sim)
        else:
            result = self._MMD_unbiased_torch(K_sim_sim, K_obs_sim)

        return result

    def _def_gaussian_kernel(self, sigma=1):
        # notice in the MMD paper they set sigma to a median value over the observation; check that.
        sigma_2 = 2 * sigma ** 2

        # def Gaussian_kernel(x, y):
        #     xy = x - y
        #     # assert np.allclose(np.dot(xy, xy), np.linalg.norm(xy) ** 2)
        #     return np.exp(- np.dot(xy, xy) / sigma_2)

        def Gaussian_kernel_vectorized(X, Y):
            """Here X and Y have shape (n_samples_x, n_features) and (n_samples_y, n_features);
            this directly computes the kernel for all pairwise components"""
            XY = X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1)  # pairwise differences
            return self.np.exp(- self.np.einsum('xyi,xyi->xy', XY, XY) / sigma_2)

        return Gaussian_kernel_vectorized

    def _compute_Gram_matrix(self, s_observations, s_simulations):

        if self.kernel_vectorized:
            K_sim_sim = self.kernel(s_simulations, s_simulations)
            K_obs_sim = self.kernel(s_observations, s_simulations)
        else:
            # this should not happen in case self.use_jax is True
            n_obs = s_observations.shape[0]
            n_sim = s_simulations.shape[0]

            K_sim_sim = np.zeros((n_sim, n_sim))
            K_obs_sim = np.zeros((n_obs, n_sim))

            for i in range(n_sim):
                # we assume the function to be symmetric; this saves some steps:
                for j in range(i, n_sim):
                    K_sim_sim[j, i] = K_sim_sim[i, j] = self.kernel(s_simulations[i], s_simulations[j])

            for i in range(n_obs):
                for j in range(n_sim):
                    K_obs_sim[i, j] = self.kernel(s_observations[i], s_simulations[j])

        return K_sim_sim, K_obs_sim

    def _MMD_unbiased(self, K_sim_sim, K_obs_sim):
        # Adapted from https://github.com/eugenium/MMD/blob/2fe67cbc7378f10f3b273cfd8d8bbd2135db5798/mmd.py
        # The estimate when distribution of x is not equal to y
        n_obs, n_sim = K_obs_sim.shape

        t_obs_sim = (2. / n_sim) * self.np.sum(K_obs_sim)
        t_sim_sim = (1. / (n_sim * (n_sim - 1))) * self.np.sum(K_sim_sim[~self.np.eye(n_sim, dtype=bool)])

        return n_obs * t_sim_sim - t_obs_sim

    def _MMD_V_estimator(self, K_sim_sim, K_obs_sim):
        # The estimate when distribution of x is not equal to y
        n_obs, n_sim = K_obs_sim.shape

        t_obs_sim = (2. / n_sim) * self.np.sum(K_obs_sim)
        t_sim_sim = (1. / (n_sim * n_sim)) * self.np.sum(K_sim_sim)

        return n_obs * t_sim_sim - t_obs_sim

    @staticmethod
    def _all_pairs(f):
        """Used to apply a function of two elements to all possible pairs."""
        f = vmap(f, in_axes=(None, 0))
        f = vmap(f, in_axes=(0, None))
        return f

    @staticmethod
    def _def_gaussian_kernel_torch(sigma=1):
        sigma_2 = 2 * sigma ** 2

        def Gaussian_kernel_vectorized(X: TensorType["batch_size", "x_size", "data_size"],
                                       Y: TensorType["batch_size", "y_size", "data_size"]) -> TensorType[
            "batch_size", "x_size", "y_size"]:
            """Here X and Y have shape (n_samples_x, n_features) and (n_samples_y, n_features);
            this directly computes the kernel for all pairwise components"""
            XY = torch.cdist(X, Y)
            return torch.exp(- torch.pow(XY, 2) / sigma_2)

        return Gaussian_kernel_vectorized

    def _compute_Gram_matrix_torch(self, simulations: TensorType["batch", "ensemble_size", "data_size"],
                                   observations: TensorType["batch", "data_size"]) -> Tuple[
            TensorType["batch", "ensemble_size", "ensemble_size"], TensorType["batch", 1, "ensemble_size"]]:

        batch_size, ensemble_size, data_size = simulations.shape

        if self.kernel_vectorized:
            observations = observations.reshape(batch_size, 1, data_size)
            K_sim_sim = self.kernel_torch(simulations, simulations)
            K_obs_sim = self.kernel_torch(observations, simulations)
        else:

            K_sim_sim = torch.zeros((batch_size, ensemble_size, ensemble_size))
            K_obs_sim = torch.zeros((batch_size, 1, ensemble_size))

            for b in range(batch_size):
                for i in range(ensemble_size):
                    # we assume the function to be symmetric; this saves some steps:
                    for j in range(i, ensemble_size):
                        K_sim_sim[b, j, i] = K_sim_sim[b, i, j] = self.kernel(simulations[b, i], simulations[b, j])

                for j in range(ensemble_size):
                    K_obs_sim[b, 0, j] = self.kernel(observations[b], simulations[b, j])

        return K_sim_sim, K_obs_sim

    @staticmethod
    def _MMD_unbiased_torch(K_sim_sim: TensorType["batch", "ensemble_size", "ensemble_size"],
                            K_obs_sim: TensorType["batch", 1, "ensemble_size"]) -> TensorType[float]:
        # Adapted from https://github.com/eugenium/MMD/blob/2fe67cbc7378f10f3b273cfd8d8bbd2135db5798/mmd.py
        # The estimate when distribution of x is not equal to y
        batch_size, ensemble_size, _ = K_sim_sim.shape

        t_obs_sim = (2. / ensemble_size) * torch.sum(K_obs_sim)

        # sum only the off-diagonal elements of K_sim_sim: first set them to 0:
        # this does not work inside automatic differentiation!
        # K_sim_sim[:, range(ensemble_size), range(ensemble_size)] = 0
        # t_sim_sim = (1. / (ensemble_size * (ensemble_size - 1))) * torch.sum(K_sim_sim)

        # alternatively, sum only the off-diagonal elements.
        off_diagonal_sum = torch.sum(
            K_sim_sim.masked_select(
                torch.stack([~torch.eye(ensemble_size, dtype=bool, device=K_sim_sim.device)] * batch_size)))
        t_sim_sim = (1. / (ensemble_size * (ensemble_size - 1))) * off_diagonal_sum

        return t_sim_sim - t_obs_sim

    @staticmethod
    def _MMD_V_estimator_torch(K_sim_sim: TensorType["batch", "ensemble_size", "ensemble_size"],
                               K_obs_sim: TensorType["batch", 1, "ensemble_size"]) -> TensorType[float]:
        # Adapted from https://github.com/eugenium/MMD/blob/2fe67cbc7378f10f3b273cfd8d8bbd2135db5798/mmd.py
        # The estimate when distribution of x is not equal to y
        batch_size, ensemble_size, _ = K_sim_sim.shape

        t_obs_sim = (2. / ensemble_size) * torch.sum(K_obs_sim)

        t_sim_sim = (1. / (ensemble_size * ensemble_size)) * torch.sum(K_sim_sim)

        return t_sim_sim - t_obs_sim
