import torch as th
from hion.types import LagrangianFunc

"""
Author: Josue N Rivera
"""

def constant(value:th.DoubleTensor = 1.0, **kargs) -> LagrangianFunc:
    return lambda time, _, __: th.zeros_like(time)

def scale(first_state_vector: th.DoubleTensor,
          first_control_vector: th.DoubleTensor, **kargs) -> LagrangianFunc:
    
    def dldx(_: th.DoubleTensor,
                   first_order_state: th.DoubleTensor, 
                   first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        return (first_order_state/first_order_state)*first_state_vector

    return dldx

def quadratic(first_state_matrix: th.DoubleTensor,
              first_control_matrix: th.DoubleTensor, **kargs) -> LagrangianFunc:
    
    matrix = first_state_matrix+first_state_matrix.T
    def dldx(_: th.DoubleTensor,
                   first_order_state: th.DoubleTensor, 
                   first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        return th.matmul(first_order_state, matrix)

    return dldx

def quadratic_error(first_state_matrix:th.DoubleTensor = th.eye(1),
                    first_control_matrix:th.DoubleTensor = th.eye(1), **kargs) -> LagrangianFunc:

    if kargs['reference_state'].size(1) != first_state_matrix.size(0):
        raise ValueError('Reference state vector must be the same size as the state vector')

    matrix = first_state_matrix+first_state_matrix.T
    def dldx(_: th.DoubleTensor,
                   first_order_state: th.DoubleTensor,
                   first_order_control: th.DoubleTensor) -> th.DoubleTensor:
        
        error = kargs['reference_state'] - first_order_state[:, kargs['reference_mask']] if type(kargs['reference_state']) != type(None) else first_order_state*0
    
        return th.matmul(error, matrix)

    return dldx