from typing import Callable, Any

import torch
import torch.nn as nn
from torch.utils._pytree import PyTree

from .utils import flatten_pytree

Array = Any

"""Code for functional-style networks in PyTorch."""

class FunctionalModel():
  """Functional wrapper around a PyTorch model.
  
  Use make_functional() to create one.
  
  Example:
    >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
    >>> flat_params, functional_model = FunctionalModel.make_functional(model)
    >>> output = functional_model.apply(flat_params, torch.randn(1, 10))
  """

  def __init__(self, module: nn.Module, unflatten_fn: Callable[[Array], PyTree]):
    """Don't use this constructor - use make_functional()
    
    Args:
      module: the PyTorch model
      unflatten_fn: unflattens a flat parameter vector into a PyTree
    """
    self._module = module
    self._unflatten_fn = unflatten_fn
  
  def apply(self, flat_params: Array, x: Array):
    """Run a forward pass of the network.
    
    Args:
      flat_params (Array): parameters, as a flat vector
      x (Array): inputs
    """
    params = self.unflatten(flat_params)
    return torch.func.functional_call(self._module, params, x)
  
  def unflatten(self, flat_params: Array) -> PyTree:
    """Unflattens a flat parameter vector into a PyTree.
    
    Args:
      flat_params (Array): a flat parameter vector
      
    Returns:
      pytree: the same parameters unflattened into a PyTree
    """
    return self._unflatten_fn(flat_params)
  
  @classmethod
  def make_functional(cls, module: nn.Module):
    """Take a PyTorch module that encodes a neural network, and make it functional.
    
    Args:
      module (nn.Module): the PyTorch module to wrap
      
    Returns:
      Array: current parameters, as a flat vector
      FunctionalModel: a functional version of the model
    """
    # TODO this may not handle batchnorm correctly - might need to collect
    # named_buffers() too and passing them to functional_call via the buffers arg
    params = {k: v.detach() for k, v in module.named_parameters()}
    flat_params, unflatten = flatten_pytree(params)
    return flat_params, FunctionalModel(module, unflatten)
