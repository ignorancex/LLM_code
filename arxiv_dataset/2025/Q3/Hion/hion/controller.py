from typing import Optional, Tuple, Union
import torch as th
import torch.nn as nn
import hion.dynamics as hdy, hion.distribution as hdi
from hion.fn import factorial, wrap_value, EPSILON, taylor_operator

"""
Author: Josue N Rivera
"""
    
class NN(nn.Module):
    def __init__(self, in_n, out_n,
                       width:int = 20,
                       depth:int = 2,
                       blocks:int = 1,
                       activation = nn.SiLU(),
                       dtype:th.dtype = th.double) -> None:
        super().__init__()

        self.entry = nn.Sequential(nn.Linear(in_n, width, dtype=dtype), nn.SiLU())

        self.exit = nn.Sequential(nn.SiLU(), nn.Linear(width, width//2, dtype=dtype), nn.SiLU(),nn.Linear(width//2, out_n, dtype=dtype))

        self.blocks = nn.ModuleList([
            nn.Sequential(*[nn.Sequential(nn.Linear(width, width, dtype=dtype), activation) for _ in range(max(0, depth))]) for __ in range(blocks)]
        )

    def forward(self, x:th.DoubleTensor) -> th.DoubleTensor:
        x = self.entry(x)
        for block in self.blocks:
            x = block(x)+x
        return self.exit(x)

class Controller(nn.Module):

    def __init__(self, distribution:hdi.StateDistribution, dynamics:hdy.Dynamics) -> None:
        super().__init__()
        self.distribution = distribution
        self.dynamics = dynamics

    def forward(self, 
                t: th.DoubleTensor, 
                x0: th.DoubleTensor, 
                xr: Optional[th.DoubleTensor] = None) -> Tuple[th.DoubleTensor, th.DoubleTensor]:
        
        """
        Returns:
            inferred state
            inferred control
            inferred costate
        """

        raise NotImplementedError

class TMano(Controller):

    def __init__(self,
                 state_nn:nn.Module,
                 costate_nn:nn.Module,
                 distribution:hdi.StateDistribution,
                 dynamics:hdy.Dynamics) -> None:
        Controller.__init__(self, distribution=distribution, dynamics=dynamics)

        self.state_nn    = state_nn
        self.costate_nn  = costate_nn

        self.reference_exist = self.distribution.reference_n > 0

        self.reference_invariant_mask = self.dynamics.first_invariant_state_mask[self.distribution.reference_mask]
        self.reference_invariant_idxs = self.reference_invariant_mask.nonzero().view(-1)

        self.invariant_reference_mask = self.distribution.reference_mask[self.dynamics.first_invariant_state_mask]

    def forward(self, 
                t: th.DoubleTensor, 
                x0: th.DoubleTensor, 
                xr: Optional[th.DoubleTensor] = None,
                return_costate = False) -> Union[Tuple[th.DoubleTensor, th.DoubleTensor], Tuple[th.DoubleTensor, th.DoubleTensor, th.DoubleTensor]]:
        
        """
        Returns:
            inferred first order state
            inferred first order control
            inferred first order costate
        """
        if not self.training:
            t = t + EPSILON # used to handle grad of pow bug #89757 (PyTorch)

        # 1. Filter for invariant state
        if self.dynamics.has_invariant_states:
            delta = x0[:, self.dynamics.first_invariant_state_mask]
            reference_delta = delta[:, self.invariant_reference_mask]

            x0 = x0.index_add(1, self.dynamics.first_invariant_state_idxs, delta, alpha=-1)
            xr = xr.index_add(1, self.reference_invariant_idxs, reference_delta, alpha=-1)

        # 2. State genetaor 
        input = th.cat([t, x0, xr], dim=1) if self.reference_exist else th.cat([t, x0], dim=1)
        state_h = self.state_nn(input)

        # 3. Apply taylor operator
        if self.dynamics.has_invariant_states:
                x0 = x0.index_add(1, self.dynamics.first_invariant_state_idxs, delta)
                xr = xr.index_add(1, self.reference_invariant_idxs, reference_delta)

        state = taylor_operator(t, x0, state_h, dynamics=self.dynamics) #requires original x0
        
        # 4. Obtain state vector
        first_state = self.dynamics.first_state_representation(t, state)
        highest_state = self.dynamics.highest_order_state(t, first_state)

        # 5. Obtain control
        control = self.dynamics.extract_u(t, first_state, highest_state)
        first_control = self.dynamics.first_control_representation(t, control)
        
        # 6. Costate genetaor
        if return_costate:
            input = th.cat([t, x0, xr, first_state, highest_state, first_control], dim=1) if self.reference_exist else th.cat([t, x0, first_state, highest_state, first_control], dim=1)

            first_costate = self.costate_nn(input)

            return first_state, first_control, first_costate
        else:
            return first_state, first_control

class PaperTmano(TMano):
    def __init__(self,
                 distribution:hdi.StateDistribution,
                 dynamics:hdy.Dynamics,
                 state_dict:dict,
                 costate_dict:dict,
                 dtype:th.dtype = th.double) -> None:

        state_size = dynamics.primitive_state_n
        first_state_size = dynamics.first_order_state_n
        first_control_size = dynamics.first_order_control_n
        costate_size = dynamics.first_order_state_n
        reference_size = distribution.reference_n

        ## NNs
        state_nn = NN(1+first_state_size+reference_size, state_size, **state_dict, dtype=dtype)

        costate_nn = NN(1+2*first_state_size+state_size+reference_size+first_control_size, costate_size, **costate_dict, dtype=dtype)
        
        TMano.__init__(self,
            state_nn     = state_nn,
            costate_nn   = costate_nn,
            distribution = distribution,
            dynamics     = dynamics
        )

class PendulumController(PaperTmano):

    def forward(self, t: th.DoubleTensor,
                x0: th.DoubleTensor,
                xr: Optional[th.DoubleTensor] = None,
                return_costate:bool = False) -> Union[Tuple[th.DoubleTensor, th.DoubleTensor], Tuple[th.DoubleTensor, th.DoubleTensor, th.DoubleTensor]]:
        
        out = list(PaperTmano.forward(t, x0, xr, return_costate))
        out[0][:, 0:1] = wrap_value(out[0][:, 0:1]) # first state

        return tuple(out)

class RobotArmController(PaperTmano):

    def forward(self, t: th.DoubleTensor,
                x0: th.DoubleTensor,
                xr: Optional[th.DoubleTensor] = None,
                return_costate=False) -> Union[Tuple[th.DoubleTensor, th.DoubleTensor], Tuple[th.DoubleTensor, th.DoubleTensor, th.DoubleTensor]]:
        
        out = list(PaperTmano.forward(t, x0, xr, return_costate))
        out[0][:, 0:1] = wrap_value(out[0][:, 0:1]) # first state
        out[0][:, 2:3] = wrap_value(out[0][:, 2:3])

        return tuple(out)

