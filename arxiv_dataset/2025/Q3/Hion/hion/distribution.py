from typing import List, Optional, Tuple, Union
import numpy as np
import torch as th
from torch.distributions.distribution import Distribution
from torch.distributions import Normal, Uniform

class ConstantDistribution(Distribution):
    def __init__(self, constant:float = 1.0, device = None, dtype=th.double):
        self.constant = constant

        if device is None:
            device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.device = device
        self.dtype = dtype

    def sample(self, sample_shape:th.Size):
        return th.ones(sample_shape, dtype=self.dtype, device=self.device)*self.constant

class StateDistribution(Distribution):

    def __init__(self, first_state_dist:List[Distribution], 
                 reference_state_g:List[Distribution] = [],
                 reference_mask: Optional[Union[List[bool], np.ndarray, th.BoolTensor]] = None,
                 device = None):
        
        if device is None:
            device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.device = device
        
        self.first_state_dist = first_state_dist
        self.reference_state_g = reference_state_g

        self.reference_mask = [True]*len(self.reference_state_g) \
                                  + [False]*(len(self.first_state_dist)-len(self.reference_state_g))\
                                  if reference_mask is None else reference_mask
        self.reference_mask = th.tensor(self.reference_mask, dtype=th.bool, device=device)
        self.reference_idxs = self.reference_mask.nonzero().view(-1)
        self.reference_n = sum(self.reference_mask)

    def sample_state(self, n:int):
        return th.cat([state_dist.sample((n, 1)) for state_dist in self.first_state_dist], dim=1)
    
    def sample_reference(self, observed_state:th.DoubleTensor):

        reference_samples = None

        if self.reference_n > 0:
            n = observed_state.shape[0]

            reference_samples = observed_state[:, self.reference_mask] \
                + th.cat([state_dist.sample((n, 1)) for state_dist in self.reference_state_g], dim=1)
            
        return reference_samples
    
    def sample(self, n: int) -> Tuple[th.DoubleTensor, th.DoubleTensor, Optional[th.DoubleTensor]]:

        observed_samples = self.sample_state(n)
        reference_samples = self.sample_reference(observed_samples)
                
        return observed_samples, reference_samples
    
class SecondOrderLinearStateDistribution(StateDistribution):

    def __init__(self, device = None):

        if device is None:
            device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        
        super().__init__(first_state_dist=[
                            Uniform(th.tensor(-5, dtype=th.double, device=device), th.tensor(5, dtype=th.double, device=device)),
                            Normal(th.tensor(0.0, dtype=th.double, device=device), th.tensor(1, dtype=th.double, device=device))
                        ],reference_state_g=[
                            Normal(th.tensor(0.0, dtype=th.double, device=device), th.tensor(1, dtype=th.double, device=device))
                        ],reference_mask=[True, False], device=device)
    
class VanDerPolStateDistribution(StateDistribution):

    def __init__(self, device = None):

        if device is None:
            device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        
        super().__init__(first_state_dist=[
                            Uniform(th.tensor(-5, dtype=th.double, device=device), th.tensor(5, dtype=th.double, device=device)),
                            Normal(th.tensor(0.0, dtype=th.double, device=device), th.tensor(1, dtype=th.double, device=device))
                        ],reference_state_g=[
                            Normal(th.tensor(0.0, dtype=th.double, device=device), th.tensor(1, dtype=th.double, device=device))
                        ],reference_mask=[True, False], device=device)
    
class VanDerPolTrackerStateDistribution(StateDistribution):

    def __init__(self, device = None):

        if device is None:
            device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        
        super().__init__(first_state_dist=[
                            Uniform(th.tensor(-5, dtype=th.double, device=device), th.tensor(5, dtype=th.double, device=device)),
                            Normal(th.tensor(0.0, dtype=th.double, device=device), th.tensor(1, dtype=th.double, device=device))
                        ],reference_state_g=[
                            Normal(th.tensor(0.0, dtype=th.double, device=device), th.tensor(1, dtype=th.double, device=device)),
                            ConstantDistribution(th.tensor(0.0, dtype=th.double, device=device))
                        ], device=device)
        
    def sample_reference(self, observed_state: th.DoubleTensor):

        n = observed_state.shape[0]
        reference_samples = observed_state[:, self.reference_mask]

        reference_samples[:, 0:1] += self.reference_state_g[0].sample((n, 1))
        reference_samples[:, 1:2]  = self.reference_state_g[1].sample((n, 1))
        
        return reference_samples
    
class PendulumDistribution(StateDistribution):

    def __init__(self, device = None):

        if device is None:
            device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        
        super().__init__(first_state_dist=[
                            Uniform(th.tensor(-th.pi, dtype=th.double, device=device), th.tensor(th.pi, dtype=th.double, device=device)),
                            Normal(th.tensor(0.0, dtype=th.double, device=device), th.tensor(2.5, dtype=th.double, device=device))
                        ],reference_state_g=[
                            Uniform(th.tensor(-th.pi, dtype=th.double, device=device), th.tensor(th.pi, dtype=th.double, device=device))
                        ],reference_mask=[True, False], device=device)

    def sample_reference(self, observed_state: th.DoubleTensor):
        n = observed_state.shape[0]    
        return self.reference_state_g[0].sample((n, 1))
    
class RobotArmDistribution(StateDistribution):

    def __init__(self, device = None):

        if device is None:
            device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        
        super().__init__(first_state_dist=[
                            Uniform(th.tensor(-th.pi, dtype=th.double, device=device), th.tensor(th.pi, dtype=th.double, device=device)),
                            Normal(th.tensor(0.0, dtype=th.double, device=device), th.tensor(2, dtype=th.double, device=device)),
                            Uniform(th.tensor(-th.pi, dtype=th.double, device=device), th.tensor(th.pi, dtype=th.double, device=device)),
                            Normal(th.tensor(0.0, dtype=th.double, device=device), th.tensor(2, dtype=th.double, device=device))
                        ],reference_state_g=[
                            Uniform(th.tensor(-th.pi, dtype=th.double, device=device), th.tensor(th.pi, dtype=th.double, device=device)),
                            Uniform(th.tensor(-th.pi, dtype=th.double, device=device), th.tensor(th.pi, dtype=th.double, device=device))
                        ], device=device)
        
    def sample_reference(self, n:int):
        return th.cat([state_dist.sample((n, 1)) for state_dist in self.reference_state_g], dim=1)
    
