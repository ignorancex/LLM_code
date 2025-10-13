from typing import Any, Protocol, Tuple

import torch
import torch.nn as nn

Array = Any

"""This file contains implementations of loss criteria."""

class LossCriterion(Protocol):
  def __call__(self, f: Array, y: Array) -> Array: ...
  """A loss criterion takes predictions f and labels y and returns a scalar."""


def mse_loss(f: Array, y: Array) -> Array:
    """Compute mean squared error between predictions and labels.
    
    Args:
      f: predictions, as an array[float] of shape [N1, N2, ..., NK]
      y: labels, as an array[float] of the same shape
      
    Returns:
      (scalar) MSE loss
    """
    return 0.5 * (f - y).square().sum(-1).mean()


def mse_categorical_loss(f: Array, y: Array, ignore_idx: int = -1) -> Array:
    """Compute mean squared error loss for a multiclass classification problem.
    
    For multiclass classification, we define MSE loss using one-hot targets.
    
    Args:
      f: predictions, as an array[float] of shape [N1, N2, .., NK, C]
      y: labels, as an array[int] of shape [N1, N2, ..., NK] filled with {0, 1, ... C-1}
      
    Returns:
      (scalar) MSE loss
    """    
    f, y = _flatten(f, y, ignore_idx=ignore_idx)

    # convert labels to one-hot
    one_hot_y = torch.nn.functional.one_hot(y).float()
    
    return mse_loss(f, one_hot_y)


def mse_binary_loss(f: Array, y: Array, ignore_idx: int = -1) -> Array:
    """Compute mean squared error loss for a binary classification problem.
    
    For binary classification, we define MSE loss using +1 / -1 targets.
    
    Args:
      f: predictions, as an array[float] of shape [N1, N2, .., NK, 1],
      y: labels, as an array[int] of shape [N1, N2, ..., NK] filled with {0, 1}
      
    Returns:
      (scalar) MSE loss
    """
    f, y = _flatten(f, y, ignore_idx=ignore_idx)
  
    # convert labels in {0, 1} to {-1.0, 1.0}
    y_binary = (2*y - 1).float()
    
    # add dimension
    y_binary = y_binary.unsqueeze(1)
        
    return mse_loss(f, y_binary)


def ce_categorical_loss(f: Array, y: Array, ignore_idx: int = -1) -> Array:
    """Compute cross entropy loss for a multiclass classification problem.
    
    Args:
      f: logits, as an array[float] of shape [N1, N2, .., NK, C]
      y: labels, as an array[int] of shape [N1, N2, ..., NK] filled with {0, 1, ... C-1}
      
    Returns:
      (scalar) cross-entropy loss
    """
    f, y = _flatten(f, y, ignore_idx=ignore_idx)
    return nn.functional.cross_entropy(f, y)


def ce_binary_loss(f: Array, y: Array, ignore_idx: int = -1) -> Array:
    """Compute binary cross entropy (i.e. logistic) loss for binary classification problem.
    
    Args:
      f: logits, as an array[float] of shape [N1, N2, .., NK, 1],
      y: labels, as an array[int] of shape [N1, N2, ..., NK] filled with {0, 1}
      
    Returns:
      (scalar) binary cross-entropy loss
    """
    f, y = _flatten(f, y, ignore_idx=ignore_idx)

    # convert label from {0, 1} to {0.0, 1.0} and add dimension
    y = y.float().unsqueeze(1)
        
    return nn.functional.binary_cross_entropy_with_logits(f, y)


def categorical_accuracy(f: Array, y: Array, ignore_idx: int =-1) -> Array:
    """Compute accuracy for a multiclass classification problem.
    
     Args:
      f: logits, as an array[float] of shape [N1, N2, .., NK, C]
      y: labels, as an array[int] of shape [N1, N2, ..., NK] filled with {0, 1, ... C-1}
      
    Returns:
      (scalar) the accuracy   
    """
    f, y = _flatten(f, y, ignore_idx=ignore_idx)
    return (y == f.argmax(-1)).float().mean()


def binary_accuracy(f: Array, y: Array, ignore_idx: int = -1) -> Array:
    """Compute accuracy for a binary classification problem.
    
    Args:
      f: predictions, as an array[float] of shape [N1, N2, .., NK, 1],
        where >0 is a prediction for one class and <0 is a prediction for the other
      y: labels, as an array[int] of shape [N1, N2, ..., NK] filled with {0, 1}
      
    Returns:
      (scalar) the accuracy
    """
    f, y = _flatten(f, y, ignore_idx=ignore_idx)
    return ((y.unsqueeze(1) == 1) == (f > 0)).float().mean()


def _flatten(f: Array, y: Array, ignore_idx=-1) -> Tuple[Array, Array]:
    """Flatten prediction/label tensors and apply masking.
    
    Given the arguments:
      f: an array of shape [N1, N2, .., NK, C]
      y: an array of shape [N1, N2, .., NK]
      
    this function first flattens f into shape [N1*N2*...*NK, C] and 
    y into shape [N1*N2*...*NK] and then removes the rows of each
    where y == ignore_index, which means they should be masked out.
    """
    nclass = f.shape[-1]

    # flatten, so that each row can be viewed as a different example
    f, y = f.view(-1, nclass), y.view(-1)
    
    # discard rows where y=ignore_idx
    keep = y != ignore_idx
    f, y = f[keep], y[keep]
    
    return f,y
    