from turtle import forward
import warnings

import numpy as np
from abcpy.continuousmodels import ProbabilisticModel, Continuous, InputConnector, Uniform, Normal
from abcpy.statistics import Statistics, Identity
from scipy.optimize import fsolve
from scipy.stats import moment, multivariate_normal, skew, kurtosis  # kurtosis computes the excess kurtosis
from statsmodels.tsa.arima_process import arma_generate_sample
import torch
from torch.distributions import multivariate_normal
from typing import List, Callable
from torchtyping import TensorType
from src.transformers import BoundedVarTransformer
from torch.nn.utils import parameters_to_vector

class mv_g_and_k(ProbabilisticModel, Continuous):
    def __init__(self, name:str='Mv G and K', seed:int=42, priors:List=None):
        self.seed = seed
        self.param_dim = 5
        self.scores = [] # for debugging purposes, just to save the score
        self.model = None

        if priors is None:
            # Use default prior parameterisation
            A = Uniform([[0], [4]], name='A')
            B = Uniform([[0], [4]], name='B')
            g = Uniform([[0], [4]], name='g')
            k = Uniform([[0], [4]], name='k')
            rho = Uniform([[-np.sqrt(3)/3], [np.sqrt(3)/3]], name='rho')
            parameters = [A,B,g,k,rho]
        else:
            paramaters = priors

        input_parameters = InputConnector.from_list(parameters)
        super(mv_g_and_k, self).__init__(input_parameters, name)

    def forward_simulate(self, input_values, num_forward_simulations,rng=np.random.RandomState()):
        # Only implemented for legacy API in w estimation
        # Does not store grads!
        # Use torch_forward_simulate
        sims = self.torch_forward_simulate(torch.tensor(input_values), num_forward_simulations)

        return [item.detach().numpy() for item in sims]

    def torch_forward_simulate(self, params, num_forward_simulations: int, seed:int=None):
        """
        params: List of three parameters 
        num_forward_simulations: int
        """
        if seed is None:
            torch.manual_seed(self.seed)
            self.seed += 1
        else:
            torch.manual_seed(seed)

        batch_size = num_forward_simulations

        self.model = torch_mv_g_and_k(*params)
        sims = self.model.forward_simulate(batch_size).reshape(num_forward_simulations, self.get_output_dimension())

        return sims

    def get_output_dimension(self):
        return 5 

    def _check_input(self, input_values):
        """
        """
        return True

    def _check_output(self, values):
        return True

class uni_g_and_k(ProbabilisticModel, Continuous):
    def __init__(self, name:str='Uni G and K', seed:int=42, priors:List=None):
        self.seed = seed
        self.param_dim = 4
        self.scores = [] # for debugging purposes, just to save the score
        self.model = None

        if priors is None:
            # Use default prior parameterisation
            A = Uniform([[0], [4]], name='A')
            B = Uniform([[0], [4]], name='B')
            g = Uniform([[0], [4]], name='g')
            k = Uniform([[0], [4]], name='k')
            parameters = [A,B,g,k]
        else:
            paramaters = priors

        input_parameters = InputConnector.from_list(parameters)
        super(uni_g_and_k, self).__init__(input_parameters, name)

    def forward_simulate(self, input_values, num_forward_simulations,rng=np.random.RandomState()):
        # Only implemented for legacy API in w estimation
        # Does not store grads!
        # Use torch_forward_simulate
        sims = self.torch_forward_simulate(torch.tensor(input_values), num_forward_simulations)

        return [item.detach().numpy() for item in sims]

    def torch_forward_simulate(self, params, num_forward_simulations: int, seed:int=None):
        """
        params: List of three parameters 
        num_forward_simulations: int
        """
        if seed is None:
            torch.manual_seed(self.seed)
            self.seed += 1
        else:
            torch.manual_seed(seed)

        batch_size = num_forward_simulations

        self.model = torch_uni_g_and_k(*params)
        sims = self.model.forward_simulate(batch_size).reshape(num_forward_simulations, self.get_output_dimension())

        return sims

    def get_output_dimension(self):
        return 1

    def _check_input(self, input_values):
        """
        """
        return True

    def _check_output(self, values):
        return True

class torch_uni_g_and_k(torch.nn.Module):
    """
    Takes as parameters (A,B,g,k) (Note, B>0 and k>-2.5)
    Takes as input random noise
    Outputs a draw from the g and k distribution
    """

    def __init__(self, A, B, g, k):
        super().__init__()
        self.c = 0.8

        self.A = A
        self.B = B
        self.g = g
        self.k = k

    def forward_simulate(self, num_forward_simulations):
        return self._draw_g_and_k(num_forward_simulations)

    def _draw_g_and_k(self, n):
        """n is the number of samples, dim is the dimensions"""
        z = torch.randn(n)
        return self._z2gk(z)

    def _z2gk(self, z):
        if self.g == 0:
            term1 = 1
        else:
            term1 = (1 + self.c * torch.tanh(self.g * z * 0.5))

        term2 = z * (1 + z ** 2) ** self.k

        return self.A + self.B * term1 * term2

class torch_mv_g_and_k(torch.nn.Module):
    """
    Takes as parameters (A,B,g,k) (Note, B>0 and k>-2.5)
    Takes as input random noise
    Outputs a draw from the g and k distribution
    """

    def __init__(self, A, B, g, k, rho):
        super().__init__()
        self.c = 0.8

        self.dim = 5
        self.A = A
        self.B = B
        self.g = g
        self.k = k
        self.rho = rho


    def forward_simulate(self, num_forward_simulations):
        return self._draw_multiv_g_and_k(num_forward_simulations, self.dim)

    def _create_cov_matrix(self, dim, rho):
        return torch.eye(dim) + rho * torch.diag(torch.ones(dim-1), diagonal=1) + rho * torch.diag(torch.ones(dim-1), diagonal=-1)

    def _draw_multiv_g_and_k(self, n, dim):
        """n is the number of samples, dim is the dimensions"""
        # define the covariance matrix first:

        cov = self._create_cov_matrix(dim, self.rho)

        z = multivariate_normal.MultivariateNormal(torch.zeros(dim), cov).rsample((n,))
        #z = rng.multivariate_normal(mean=np.zeros(dim), cov=cov, size=n)
        return self._z2gk(z)

    def _z2gk(self, z):
        if self.g == 0:
            term1 = 1
        else:
            term1 = (1 + self.c * torch.tanh(self.g * z * 0.5))

        term2 = z * (1 + z ** 2) ** self.k

        return self.A + self.B * term1 * term2
