import numpy as np
from abcpy.continuousmodels import ProbabilisticModel, Continuous, InputConnector, Uniform, Normal
import torch
import torchsde
from torchsde import BrownianInterval
from einops import rearrange
from torchtyping import TensorType
from src.transformers import BoundedVarTransformer
from typing import List

class Lorenz96SDE(ProbabilisticModel, Continuous):
    def __init__(self, 
                 name:str='Lorenz96SDE', 
                 seed:int=42, 
                 simulation_method='euler', 
                 ts=torch.linspace(0, 1.5, 21), 
                 priors:List=None, 
                 y0=torch.tensor([6.4558, 1.1054, -1.4502, -0.1985, 1.1905, 2.3887, 5.6689, 6.7284],dtype=torch.float64)
                 ):
        self.seed = seed
        self.param_dim = 3
        self.ts = ts
        self.simulation_method = simulation_method
        self.scores = [] # for debugging purposes, just to save the score
        self.model = None
        self.y0 = y0

        if priors is None:
            # Use default prior parameterisation
            b0 = Uniform([[1.4], [2.2]], name='b0')
            b1 = Uniform([[0], [1]], name='b1')
            sigma = Uniform([[1.5], [2.5]], name='sigma')
            parameters = [b0, b1, sigma]
        else:
            paramaters = priors

        input_parameters = InputConnector.from_list(parameters)
        super(Lorenz96SDE, self).__init__(input_parameters, name)

    def forward_simulate(self, input_values, num_forward_simulations,rng=np.random.RandomState()):
        # Not implemented
        # Only implemented for legacy API in w estimation
        # Does not store grads!
        # Use torch_forward_simulate
        #sims = self.torch_forward_simulate(torch.tensor(input_values), num_forward_simulations)
        #return [item.detach().numpy() for item in sims]
        pass

    def torch_forward_simulate(self, params, num_forward_simulations: int, seed:int=None, normalise=True):
        """
        params: List of three parameters 
        num_forward_simulations: int
        """
        if seed is None:
            torch.manual_seed(self.seed)
            seed = self.seed
            self.seed += 1
        else:
            torch.manual_seed(seed)

        batch_size = num_forward_simulations

        self.model = torch_lorenz96(*params)

        y0 = self.y0.repeat(batch_size, 1)
        bm = BrownianInterval(t0=self.ts[0], 
                      t1=self.ts[-1], 
                      size=(batch_size, 8), entropy=seed)
        ys = torchsde.sdeint(self.model, y0, self.ts, method=self.simulation_method, bm=bm)  # ys will have shape (t_size, batch_size, state_size)
        if normalise:
            ys = (ys - y0.mean()) / y0.std()
        ys = ys[1:]
        ys = rearrange(ys, 't b d -> b (t d)')  # in this way return 1d arrays as outputs

        return ys

    def get_output_dimension(self):
        return 160  # 8 * 20

    def _check_input(self, input_values):
        """
        """
        # todo do you need sigma>0 here?
        return True

    def _check_output(self, values):
        return True

class torch_lorenz96(torch.nn.Module):
    """
    Model for Stochastic Lorenz96
    To initialise, pass in the parameters b0, b1, sigma 
    """
    def __init__(self, b0, b1, sigma):
        super().__init__()
        self.b0 = b0
        self.b1 =  b1
        self.sigma = sigma

        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def _torch_l96(self, x, f):

        """"
        takes an input a tensor x of shape (batch_size, shape_size/param_size) 
        This computes the time derivative for the non-linear deterministic Lorenz 96 Model of arbitrary dimension n.
        dx/dt = f(x) 
        """

        # shift minus and plus indices
        x_m_2 = torch.cat([x[:,-2:], x[:,:-2]],1)
        x_m_1 = torch.cat([x[:,-1:], x[:,:-1]],1)
        x_p_1 = torch.cat((x[:,1:], x[:,0:1]),1)

        dxdt = (x_p_1-x_m_2)*x_m_1 - x + f

        return dxdt

    def f(self, t, y):
        return self._torch_l96(y,torch.tensor(10)) - self.b0 - self.b1 * y 
    
    def g(self, t, y):
        return self.sigma * torch.ones(y.size())