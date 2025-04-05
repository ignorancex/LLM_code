"""The Fast Gradient Method attack."""
from typing import Iterable
import numpy as np
import torch

#from cleverhans.future.torch.utils import optimize_linear

def optimize_linear(grad, eps, norm=np.inf):
  """
  Solves for the optimal input to a linear function under a norm constraint.

  Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

  :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
  :param eps: float. Scalar specifying size of constraint region
  :param norm: np.inf, 1, or 2. Order of norm constraint.
  :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
  """

  red_ind = list(range(1, len(grad.size())))
  avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
  if norm == np.inf:
    # Take sign of gradient
    optimal_perturbation = torch.sign(grad)
  elif norm == 1:
    abs_grad = torch.abs(grad)
    sign = torch.sign(grad)
    red_ind = list(range(1, len(grad.size())))
    abs_grad = torch.abs(grad)
    ori_shape = [1]*len(grad.size())
    ori_shape[0] = grad.size(0)

    max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
    max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
    num_ties = max_mask
    for red_scalar in red_ind:
      num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
    optimal_perturbation = sign * max_mask / num_ties
    # TODO integrate below to a test file
    # check that the optimal perturbations have been correctly computed
    opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
    assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
  elif norm == 2:
    square = torch.max(
        avoid_zero_div,
        torch.sum(grad ** 2, red_ind, keepdim=True)
        )
    optimal_perturbation = grad / torch.sqrt(square)
    # TODO integrate below to a test file
    # check that the optimal perturbations have been correctly computed
    opt_pert_norm = optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
    one_mask = (
        (square <= avoid_zero_div).to(torch.float) * opt_pert_norm +
        (square > avoid_zero_div).to(torch.float))
    assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
  else:
    raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                              "currently implemented.")

  # Scale perturbation to be the solution for the norm=eps rather than
  # norm=1 problem
  scaled_perturbation = eps * optimal_perturbation
  return scaled_perturbation



def fast_gradient_method(model_fn, x, eps, norm, loss_fn=None,
                         clip_min=None, clip_max=None, y=None, targeted=False, sanity_checks=False):
  """
  PyTorch implementation of the Fast Gradient Method.
  :param model_fn: a callable that takes an input tensor and returns the model logits.
  :param x: input tensor.
  :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
  :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
  :param clip_min: (optional) float. Minimum float value for adversarial example components.
  :param clip_max: (optional) float. Maximum float value for adversarial example components.
  :param y: (optional) Tensor with true labels. If targeted is true, then provide the
            target label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting adversarial samples. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None.
  :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
  :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
            memory or for unit tests that intentionally pass strange input)
  :return: a tensor for the adversarial example
  """
  if norm not in [np.inf, 1, 2]:
    raise ValueError("Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm))
  if isinstance(eps, float):
    if eps < 0:
      raise ValueError("eps must be greater than or equal to 0, got {} instead".format(eps))
    if eps == 0:
      return x
  else:
    if not isinstance(eps, list): # list of callables
      if torch.all(eps < 0):
        raise ValueError("eps must be greater than or equal to 0, got {} instead".format(eps))
      if torch.all(eps == 0):
        return x

  if clip_min is not None and clip_max is not None:
    if isinstance(clip_min, Iterable):
      pass
    else:
      if clip_min > clip_max:
        raise ValueError(
            "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                clip_min, clip_max))

  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    assert_ge = torch.all(torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype)))
    asserts.append(assert_ge)

  if clip_max is not None:
    assert_le = torch.all(torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype)))
    asserts.append(assert_le)

  # x needs to be a leaf variable, of floating point type and have requires_grad being True for
  # its grad to be computed and stored properly in a backward call
  x = x.clone().detach().to(torch.float).requires_grad_(True)
  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    _, y = torch.max(model_fn(x), 1)

  # Compute loss
  if len(y.shape) == 2:
    # for smoothed ground truth
    loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
  elif loss_fn is None:
    loss_fn = torch.nn.CrossEntropyLoss()
  loss = loss_fn(model_fn(x), y)
  # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
  if targeted:
    loss = -loss

  # Define gradient of loss wrt input
  loss.backward()
  optimal_perturbation = optimize_linear(x.grad, eps, norm)

  # Add perturbation to original example to obtain adversarial example
  adv_x = x + optimal_perturbation

  # If clipping is needed, reset all values outside of [clip_min, clip_max]
  if (clip_min is not None) or (clip_max is not None):
    if clip_min is None or clip_max is None:
      raise ValueError(
          "One of clip_min and clip_max is None but we don't currently support one-sided clipping")
    if isinstance(clip_min, Iterable):
      for i in range(3):
        adv_x[:, i, :, :] = torch.clamp(adv_x[:, i, :, :], clip_min[i], clip_max[i])
    else:
      adv_x = torch.clamp(adv_x, clip_min, clip_max)

  if sanity_checks:
    assert np.all(asserts)
  return adv_x
