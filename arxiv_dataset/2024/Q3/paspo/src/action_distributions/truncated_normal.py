# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# from https://github.com/toshas/torch_truncnorm

import math
from numbers import Number
from typing import Union

import torch
from torch.distributions import constraints, Distribution
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """Truncated Standard Normal distribution.

    Source: https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }
    has_rsample = True
    eps = 1e-6

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(
            batch_shape, validate_args=validate_args
        )
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        if any(
            (self.a >= self.b)
            .view(
                -1,
            )
            .tolist()
        ):
            raise ValueError("Incorrect truncation range")
        eps = self.eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp(eps, 1 - eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (
            self._little_phi_b * little_phi_coeff_b
            - self._little_phi_a * little_phi_coeff_a
        ) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = (
            1
            - self._lpbb_m_lpaa_d_Z
            - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        )
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x**2) * 0.5).exp() * CONST_INV_SQRT_2PI

    def _big_phi(self, x):
        phi = 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())
        return phi.clamp(self.eps, 1 - self.eps)

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        y = self._big_phi_a + value * self._Z
        y = y.clamp(self.eps, 1 - self.eps)
        return self._inv_big_phi(y)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5

    def rsample(self, sample_shape=None):
        if sample_shape is None:
            sample_shape = torch.Size([])
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(
            self._dtype_min_gt_0, self._dtype_max_lt_1
        )
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """Truncated Normal distribution.

    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None):
        scale = scale.clamp_min(self.eps)
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        self._non_std_a = a
        self._non_std_b = b
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale**2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        sample = self._from_std_rv(super().icdf(value))

        # clamp data but keep gradients
        sample_clip = torch.stack(
            [sample.detach(), self._non_std_a.detach().expand_as(sample)], 0
        ).max(0)[0]
        sample_clip = torch.stack(
            [sample_clip, self._non_std_b.detach().expand_as(sample)], 0
        ).min(0)[0]
        sample.data.copy_(sample_clip)
        return sample

    def log_prob(self, value):
        value = self._to_std_rv(value)
        return super(TruncatedNormal, self).log_prob(value) - self._log_scale
    





#from https://github.com/pytorch/rl/blob/main/torchrl/modules/distributions/continuous.py

class TruncatedNormalLocScale(torch.distributions.Independent):
    """Implements a Truncated Normal distribution with location scaling.

    Location scaling prevents the location to be "too far" from 0, which ultimately
    leads to numerically unstable samples and poor gradient computation (e.g. gradient explosion).
    In practice, the location is computed according to

        .. math::
            loc = tanh(loc / upscale) * upscale.

    This behaviour can be disabled by switching off the tanh_loc parameter (see below).


    Args:
        loc (torch.Tensor): normal distribution location parameter
        scale (torch.Tensor): normal distribution sigma parameter (squared root of variance)
        upscale (torch.Tensor or number, optional): 'a' scaling factor in the formula:

            .. math::
                loc = tanh(loc / upscale) * upscale.

            Default is 5.0

        min (torch.Tensor or number, optional): minimum value of the distribution. Default = -1.0;
        max (torch.Tensor or number, optional): maximum value of the distribution. Default = 1.0;
        tanh_loc (bool, optional): if ``True``, the above formula is used for
            the location scaling, otherwise the raw value is kept.
            Default is ``False``;
    """

    num_params: int = 2

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.greater_than(1e-6),
    }

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        upscale: Union[torch.Tensor, float] = 5.0,
        min: Union[torch.Tensor, float] = -1.0,
        max: Union[torch.Tensor, float] = 1.0,
        tanh_loc: bool = False,
    ):
        err_msg = "TanhNormal max values must be strictly greater than min values"
        if isinstance(max, torch.Tensor) or isinstance(min, torch.Tensor):
            if not (max > min).all():
                raise RuntimeError(err_msg)
        elif isinstance(max, Number) and isinstance(min, Number):
            if not max > min:
                raise RuntimeError(err_msg)
        else:
            if not all(max > min):
                raise RuntimeError(err_msg)

        if isinstance(max, torch.Tensor):
            self.non_trivial_max = (max != 1.0).any()
        else:
            self.non_trivial_max = max != 1.0

        if isinstance(min, torch.Tensor):
            self.non_trivial_min = (min != -1.0).any()
        else:
            self.non_trivial_min = min != -1.0
        self.tanh_loc = tanh_loc

        self.device = loc.device
        self.upscale = (
            upscale
            if not isinstance(upscale, torch.Tensor)
            else upscale.to(self.device)
        )

        if isinstance(max, torch.Tensor):
            max = max.to(self.device)
        else:
            max = torch.as_tensor(max, device=self.device)
        if isinstance(min, torch.Tensor):
            min = min.to(self.device)
        else:
            min = torch.as_tensor(min, device=self.device)
        self.min = min
        self.max = max
        self.update(loc, scale)

    def update(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        if self.tanh_loc:
            loc = (loc / self.upscale).tanh() * self.upscale
        if self.non_trivial_max or self.non_trivial_min:
            loc = loc + (self.max - self.min) / 2 + self.min
        self.loc = loc
        self.scale = scale

        base_dist = TruncatedNormal(
            loc, scale, self.min.expand_as(loc), self.max.expand_as(scale)
        )
        super().__init__(base_dist, 1, validate_args=False)

    @property
    def mode(self):
        m = self.base_dist.loc
        a = self.base_dist._non_std_a + self.base_dist._dtype_min_gt_0
        b = self.base_dist._non_std_b - self.base_dist._dtype_min_gt_0
        m = torch.min(torch.stack([m, b], -1), dim=-1)[0]
        return torch.max(torch.stack([m, a], -1), dim=-1)[0]

    def log_prob(self, value, **kwargs):
        a = self.base_dist._non_std_a + self.base_dist._dtype_min_gt_0
        a = a.expand_as(value)
        b = self.base_dist._non_std_b - self.base_dist._dtype_min_gt_0
        b = b.expand_as(value)
        value = torch.min(torch.stack([value, b], -1), dim=-1)[0]
        value = torch.max(torch.stack([value, a], -1), dim=-1)[0]
        return super().log_prob(value, **kwargs)
    


#def inv_softplus(bias: float | torch.Tensor) -> float | torch.Tensor:
def inv_softplus(bias):
    """Inverse softplus function.

    Args:
        bias (float or tensor): the value to be softplus-inverted.
    """
    is_tensor = True
    if not isinstance(bias, torch.Tensor):
        is_tensor = False
        bias = torch.tensor(bias)
    out = bias.expm1().clamp_min(1e-6).log()
    if not is_tensor and out.numel() == 1:
        return out.item()
    return out


class biased_softplus(torch.nn.Module):
    """A biased softplus module.

    The bias indicates the value that is to be returned when a zero-tensor is
    passed through the transform.

    Args:
        bias (scalar): 'bias' of the softplus transform. If bias=1.0, then a _bias shift will be computed such that
            softplus(0.0 + _bias) = bias.
        min_val (scalar): minimum value of the transform.
            default: 0.1
    """

    def __init__(self, bias: float, min_val: float = 0.01) -> None:
        super().__init__()
        self.bias = inv_softplus(bias - min_val)
        self.min_val = min_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x + self.bias) + self.min_val
