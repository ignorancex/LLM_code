from typing import List, Optional, Union
import torch as th
import torch.nn.functional as F
from hion.types import LagrangianFunc
import hion.controller as controller, hion.distribution as distribution, hion.dynamics as dynamics

"""
Author: Josue N Rivera
"""

def hamiltonian(time: th.DoubleTensor,
                first_order_state: th.DoubleTensor,
                first_order_control: th.DoubleTensor,
                costate: th.DoubleTensor,
                lagrantian:LagrangianFunc, 
                dynamics:dynamics.Dynamics) -> th.DoubleTensor:
    
    differerential = dynamics.f(time, first_order_state, first_order_control)

    return lagrantian(time, first_order_state, first_order_control) + (costate*differerential).sum(dim=1, keepdim=True)

def dhdx(time: th.DoubleTensor,
         first_order_state: th.DoubleTensor,
         first_order_control: th.DoubleTensor,
         costate: th.DoubleTensor,
         dldx:LagrangianFunc, 
         dynamics:dynamics.Dynamics) -> th.DoubleTensor:
    
    return dldx(time, first_order_state, first_order_control) + th.einsum('bm,bmn->bn', costate, dynamics.dfdx(time, first_order_state, first_order_control))

def dhdu(time: th.DoubleTensor,
                first_order_state: th.DoubleTensor,
                first_order_control: th.DoubleTensor,
                costate: th.DoubleTensor,
                dldu:LagrangianFunc, 
                dynamics:dynamics.Dynamics) -> th.DoubleTensor:
    
    return dldu(time, first_order_state, first_order_control) + th.einsum('bm,bmn->bn', costate, dynamics.dfdu(time, first_order_state, first_order_control))

def zero_step_loss(time_sample,
                first_control,
                initial_state,
                reference_sample,
                controller:controller.Controller,
                dynamics:dynamics.Dynamics,
                n = 10,
                step_size = 0.005):
    
    cum_observed_state = []
    cum_first_state = []

    observed_state = initial_state.detach()

    for _ in range(n):

        # estimate next state
        first_state, first_control_temp = controller(time_sample + step_size, initial_state, reference_sample, return_costate=False)
        cum_first_state.append(first_state[:, dynamics.state_primitive_mask])

        # Determine next observed state with current control
        observed_state = observed_state + step_size * dynamics.f(time_sample.detach(), observed_state, first_control.detach())
        cum_observed_state.append(observed_state[:, dynamics.state_primitive_mask])

        # Update next step control
        first_control = first_control_temp
        time_sample = time_sample + step_size

    return F.mse_loss(th.cat(cum_first_state, dim=0), th.cat(cum_observed_state, dim=0))

def boundary_loss(first_order_state: th.DoubleTensor,
                  bd_first_order_state: th.DoubleTensor) -> th.DoubleTensor:

    return F.mse_loss(first_order_state, bd_first_order_state)

def boundary_reference_loss(first_order_state: th.DoubleTensor,
                  bd_first_order_state: th.DoubleTensor,
                  reference_mask:Optional[th.BoolTensor] = None) -> th.DoubleTensor:
    
    first_order_state = first_order_state if reference_mask is None else first_order_state[:, reference_mask]
    return boundary_loss(first_order_state, bd_first_order_state)

def dynamics_loss(time: th.DoubleTensor,
                  first_order_state: th.DoubleTensor, 
                  first_order_control: th.DoubleTensor,
                  dynamics:dynamics.Dynamics) -> th.DoubleTensor:

    return th.mean(dynamics.ode_residual(time=time,
                                         first_order_state=first_order_state,
                                         first_order_control=first_order_control)**2)

def hamiltonian_loss(time: th.DoubleTensor,
                     first_order_state: th.DoubleTensor,
                     first_order_control: th.DoubleTensor,
                     costate: th.DoubleTensor,
                     dldu:LagrangianFunc, 
                     dynamics:dynamics.Dynamics) -> th.DoubleTensor:

    duhamiltonian = dhdu(time, first_order_state, first_order_control, costate, dldu, dynamics)
    
    return th.mean(duhamiltonian**2)

def costate_loss(time: th.DoubleTensor,
                 first_order_state: th.DoubleTensor,
                 first_order_control: th.DoubleTensor,
                 costate: th.DoubleTensor,
                 dldx:LagrangianFunc, 
                 dynamics:dynamics.Dynamics) -> th.DoubleTensor:
    
    dtcostate = th.cat(
        [th.autograd.grad(costate[:, i].sum(), time, create_graph=True)[0] for i in range(costate.size(1))],
    dim = 1)

    dxhamiltonian = dhdx(time.detach(), first_order_state, first_order_control, costate, dldx, dynamics)

    return F.mse_loss(dtcostate, -dxhamiltonian)

def costate_terminal_loss(costate: th.DoubleTensor,
                          reference_mask: Union[List[bool], th.Tensor]) -> th.DoubleTensor:
    
    if sum(reference_mask) == len(reference_mask): 
        return th.sum(costate*0.0)
    
    return th.mean(costate[:, ~reference_mask]**2)