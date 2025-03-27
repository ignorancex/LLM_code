import numpy as np
from abcpy.continuousmodels import ProbabilisticModel, Continuous, InputConnector, Uniform, MultivariateNormal
import torch
import torchsde
from torchsde import BrownianInterval
from einops import rearrange
from typing import List
from functorch import make_functional

class NeuralLorenz96SDE(ProbabilisticModel, Continuous):
    def __init__(self, 
                 name:str='Lorenz96SDE', 
                 seed:int=42, 
                 simulation_method='euler', 
                 ts=torch.linspace(0, 1.5, 21), 
                 priors:List=None, 
                 y0=torch.tensor([6.4558, 1.1054, -1.4502, -0.1985, 1.1905, 2.3887, 5.6689, 6.7284], dtype=torch.float64)
                 ):
        self.seed = seed
        self.param_dim = 111
        self.ts = ts
        self.simulation_method = simulation_method
        self.scores = [] # for debugging purposes, just to save the score
        self.model = None
        self.y0 = y0

        if priors is None:
            sigma = Uniform([[1.5], [2.5]], name='sigma')
            mlp_param = MultivariateNormal([np.ones(self.param_dim - 1).tolist(), np.eye(self.param_dim - 1).tolist()], name='mlp_param')
            parameters = [sigma, mlp_param]
        else:
            paramaters = priors

        input_parameters = InputConnector.from_list(parameters)
        super(NeuralLorenz96SDE, self).__init__(input_parameters, name)

    def forward_simulate(self, input_values, num_forward_simulations,rng=np.random.RandomState()):
        # Only implemented for legacy API in w estimation
        # Does not store grads!
        # Use torch_forward_simulate
        sims = self.torch_forward_simulate(torch.tensor(input_values), num_forward_simulations)

        return [item.detach().numpy() for item in sims]

    def torch_forward_simulate(self, params, num_forward_simulations: int, seed:int = None, normalise=True):
        """
        params : Two parameter list, first being the sigma and the second being the MLP parameters
        """
        if seed is None:
            torch.manual_seed(self.seed)
            self.seed += 1
        else:
            torch.manual_seed(seed)

        batch_size = num_forward_simulations

        self.model = neural_torch_lorenz96(params) 

        y0 = self.y0.repeat(batch_size, 1)
        bm = BrownianInterval(t0=self.ts[0], 
                      t1=self.ts[-1], 
                      size=(batch_size, 8), entropy=self.seed)
        ys = torchsde.sdeint(self.model, y0, self.ts, method=self.simulation_method,bm=bm)  # ys will have shape (t_size, batch_size, state_size)
        if normalise:
            ys = (ys - y0.mean()) / y0.std()
        ys = ys[1:]
        ys = rearrange(ys, 't b d -> b (t d)')  # in this way return 1d arrays as outputs
        
        return ys

    def get_default_mlp_params(self, seed:int = None):
        """
        Returns the default MLP parameters initialised in PyTorch
        Should make this explicit eventually as the defaults may change
        """
        if seed is None:
            torch.manual_seed(self.seed)
            self.seed += 1
        else:
            torch.manual_seed(seed)
        nn = MLP(8, 8, 6)
        _, params = make_functional(nn)

        return params

    def get_output_dimension(self):
        return 160  # 8 * 20

    def _check_input(self, input_values):
        """
        """
        # todo do you need sigma>0 here?
        return True

    def _check_output(self, values):
        return True

class neural_torch_lorenz96(torch.nn.Module):
    def __init__(self, param_list):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.mlp = MLP(8, 8, 6)
        self.mlp_func, _ = make_functional(self.mlp)
        self.set_parameters(param_list)

    def set_parameters(self, param_list):
        self.sigma = param_list[0]
        self.mlp_params = param_list[1]

    def _torch_l96(self, x, f):

        """"
        takes an input a tensor x of shape (batch_size, shape_size/param_size) 
        This computes the time derivative for the non-linear deterministic Lorenz 96 Model of arbitrary dimension n.
        dx/dt = f(x) 
        """

        # shift minus and plus indices
        x_m_2 = torch.cat([x[:,-2:], x[:,:-2]],1)
        #print(x_m_2)
        x_m_1 = torch.cat([x[:,-1:], x[:,:-1]],1)
        #print(x_m_1)
        x_p_1 = torch.cat((x[:,1:], x[:,0:1]),1)
        #print(x_p_1)

        dxdt = (x_p_1-x_m_2)*x_m_1 - x + f

        return dxdt

    def f(self, t, y):
        return self._torch_l96(y,torch.tensor(10)) - self.mlp_func(self.mlp_params, y) 
    
    def g(self, t, y):
        return self.sigma * torch.ones(y.size())


class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size):
        super().__init__()

        self._model = torch.nn.Sequential(
            torch.nn.Linear(in_size, mlp_size),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_size, out_size)
        )

    def forward(self, x):
        return self._model(x)