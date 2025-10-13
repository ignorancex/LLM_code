import torch as th
from hion.types import LagrangianFunc

"""
Author: Josue N Rivera
"""

def constant(value:th.DoubleTensor = 1.0, **kargs) -> LagrangianFunc:
    return lambda time, _, __: th.zeros_like(time)

def scale(first_state_vector: th.DoubleTensor,
          first_control_vector: th.DoubleTensor, **kargs) -> LagrangianFunc:
    
    def dldu(_: th.DoubleTensor,
                   first_order_state: th.DoubleTensor, 
                   first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        return (first_order_control/first_order_control)*first_control_vector

    return dldu

def quadratic(first_state_matrix: th.DoubleTensor,
              first_control_matrix: th.DoubleTensor, **kargs) -> LagrangianFunc:
    
    matrix = first_control_matrix+first_control_matrix.T
    def dldu(_: th.DoubleTensor,
                   first_order_state: th.DoubleTensor, 
                   first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        return th.matmul(first_order_control, matrix)

    return dldu

def quadratic_error(first_state_matrix: th.DoubleTensor,
              first_control_matrix: th.DoubleTensor, **kargs) -> LagrangianFunc:
    
    matrix = first_control_matrix+first_control_matrix.T
    def dldu(_: th.DoubleTensor,
                   first_order_state: th.DoubleTensor, 
                   first_order_control: th.DoubleTensor) -> th.DoubleTensor:

        return th.matmul(first_order_control, matrix)

    return dldu