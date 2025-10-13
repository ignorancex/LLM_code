import torch as th
from hion.types import LagrangianFunc

"""
Author: Josue N Rivera
"""

def constant(value:th.DoubleTensor = 1.0, **kargs) -> LagrangianFunc:

    return lambda time, _, __: value*th.ones_like(time)

def scale(first_state_vector: th.DoubleTensor,
          first_control_vector: th.DoubleTensor, **kargs) -> LagrangianFunc:
    
    def lagrangian(_: th.DoubleTensor,
                   first_order_state: th.DoubleTensor, 
                   first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        return (first_order_state*first_state_vector).sum(1, keepdim=True) + (first_order_control*first_control_vector).sum(1, keepdim=True)

    return lagrangian

def quadratic(first_state_matrix: th.DoubleTensor,
              first_control_matrix: th.DoubleTensor, **kargs) -> LagrangianFunc:
    
    def lagrangian(_: th.DoubleTensor,
                   first_order_state: th.DoubleTensor, 
                   first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        return (first_order_state*th.matmul(first_order_state, first_state_matrix)).sum(1, keepdim=True) + (first_order_control*th.matmul(first_order_control, first_control_matrix)).sum(1, keepdim=True)

    return lagrangian

def quadratic_error(first_state_matrix:th.DoubleTensor = th.eye(1),
                    first_control_matrix:th.DoubleTensor = th.eye(1), **kargs) -> LagrangianFunc:

    if kargs['reference_state'].size(1) != first_state_matrix.size(0):
        raise ValueError('Reference state vector must be the same size as the state vector')

    def lagrangian(_: th.DoubleTensor,
                   first_order_state: th.DoubleTensor,
                   first_order_control: th.DoubleTensor) -> th.DoubleTensor:
        
        error = kargs['reference_state'] - first_order_state[:, kargs['reference_mask']] if type(kargs['reference_state']) != type(None) else first_order_state*0
    
        return (error*th.matmul(error, first_state_matrix)).sum(1, keepdim=True) + (first_order_control*th.matmul(first_order_control, first_control_matrix)).sum(1, keepdim=True)

    return lagrangian

if __name__ == "__main__":

    lagrangian:LagrangianFunc = quadratic(state_matrix=th.diag(th.tensor([1, 1], dtype=th.double)),
                                          control_matrix=th.eye(0.1, dtype=th.double))