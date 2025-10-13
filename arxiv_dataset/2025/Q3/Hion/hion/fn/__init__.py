import torch as th
import hion.fn.lagrangian as lagrangian
import hion.fn.dldu as dldu
import hion.fn.dldx as dldx

from hion.dynamics import Dynamics

"""
Author: Josue N Rivera
"""

EPSILON = 1e-10

def sawtooth_wave(t, period = 0.5, amplitude = 1.0, phase = 0.0, displacement = 0.0):
    x = (t + phase) / period
    return amplitude * (x - th.floor(x)) + displacement

def factorial(x):
    return th.exp(th.lgamma(x+1))

def wrap_value(angle, period=2*th.pi, offset=-th.pi):
    """grad = 1.0"""
    return th.remainder(angle - offset, period) + offset

def taylor_operator(t, x0, state_h, dynamics:Dynamics):

    state = x0[:, dynamics.state_primitive_mask]
    taylor_state = factorial(dynamics.first_state_orders)*x0*(t**dynamics.first_state_orders)

    for order in th.arange(1, dynamics.highest_state_order+1):
        state = state.index_add(1, dynamics.in_state_order_idxs[order], taylor_state[:, dynamics.first_state_orders == order])

    return state + state_h*th.pow(t, dynamics.state_derivative_orders+1)