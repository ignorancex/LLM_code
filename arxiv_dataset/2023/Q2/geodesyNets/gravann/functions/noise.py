from typing import Callable

import torch.distributions
from torch import Tensor


def gaussian_noise(
        label_fn: Callable[[Tensor], Tensor],
        mean: float = 0.0,
        std: float = 1.0
) -> Callable[[Tensor], Tensor]:
    """Adds Gaussian noise to a label function.

    Args:
        label_fn: function taking a tensor and producing the labels, the base function on which the noise is added
        mean: mean value of the gaussian
        std: std value of the gaussian

    Returns:
        function taking an input tensor and evaluating labels + adding noise on the results

    """
    return lambda points_tensor: \
        label_fn(points_tensor) + torch.distributions.Normal(mean, std).sample(points_tensor.shape)


def adaptive_gaussian_noise(
        label_fn: Callable[[Tensor], Tensor],
        std: float = 1.0
) -> Callable[[Tensor], Tensor]:
    """Adds adaptive Gaussian noise to a label function. The std is multiplied with the input_points to be in the
    same order of magnitude. E.g. a std of 0.01 would lead to distortions beginning from the second digit behind the
    point.

    Args:
        label_fn: function taking a tensor and producing the labels, the base function on which the noise is added
        std: std value of the relative gaussian

    Returns:
        function taking an input tensor and evaluating labels + adding noise on the results

    """
    return lambda points_tensor: \
        torch.distributions.Normal(1.0, std).sample(points_tensor.shape) * label_fn(points_tensor)


def constant_bias(
        label_fn: Callable[[Tensor], Tensor],
        bias: Tensor = torch.tensor([0, 1, 0])
) -> Callable[[Tensor], Tensor]:
    """Adds a constant bias noise to a label function.

    Args:
        label_fn: function taking a tensor and producing the labels, the base function on which the noise is added
        bias: Adds a constant bias on each result of the label function, e.g. Tensor[0, 1, 0]

    Returns:
        function taking an input tensor and evaluating labels + adding noise on the results

    """
    if not torch.is_tensor(bias):
        bias = torch.tensor(bias)
    return lambda points_tensor: label_fn(points_tensor) + bias.to(points_tensor.get_device())


def combined_noise(
        label_fn: Callable[[Tensor], Tensor],
        mean: float = 0.0,
        std: float = 1.0,
        bias: Tensor = torch.tensor([0, 1, 0])
) -> Callable[[Tensor], Tensor]:
    """Adds a constant bias to a given label function and afterwards apply a Gaussian noise to the result.
    This simulates that a bias like solar radiation pressure is noise affecting the spacecraft whereas Gaussian noise
    is just the result of then measuring.

    Args:
        label_fn: function taking a tensor and producing the labels, the base function on which the noise is added
        mean: mean value of the gaussian
        std: std value of the gaussian
        bias: Adds a constant bias on each result of the label function, e.g. Tensor[0, 1, 0]

    Returns:
        function taking an input tensor and evaluating labels + adding noise on the results
    """
    return gaussian_noise(constant_bias(label_fn, bias), mean, std)
