import torch
from packaging import version
from torch import autograd


#the reason for the list distribution is that we can have multiple distributions for each sample which sometimes helps a lot with the numerical stability
class ListDistribution(torch.distributions.Distribution):
    has_rsample = True

    def __init__(self, distributions, sum=False):
        self.distributions = distributions
        self.sum = sum
        self._did_sample = False

    def sample(self, *args, **kwargs):
        if self._did_sample:
            raise RuntimeError("Do not create new samples from ListDistribution (sampling requires the autoregressive model to be run again)")
        
        self._did_sample = True
        
        return torch.stack([d.sample(*args, **kwargs) for d in self.distributions], dim=-1)

    def rsample(self, *args, **kwargs):
        if self._did_sample:
            raise RuntimeError("Do not create new samples from ListDistribution (sampling requires the autoregressive model to be run again)")
        
        self._did_sample = True
        
        return torch.stack([d.rsample(*args, **kwargs) for d in self.distributions], dim=-1)
    
    def log_prob(self, value):
        p = torch.stack([d.log_prob(value[..., i]) for i, d in enumerate(self.distributions)], dim=-1)

        if self.sum:
            return p.sum(dim=-1)
        else:
            return p
        
    def entropy(self) -> torch.Tensor:
        if self.sum:
            return torch.stack([d.entropy() for d in self.distributions], dim=-1).sum(dim=-1)
        else:
            return torch.stack([d.base_dist.entropy() + d.transforms[-1].scale.abs().log() for d in self.distributions], dim=-1) #TODO this only works for affine transforms!!
            
            
class FixedValueDistribution(torch.distributions.Distribution):
    def __init__(self, value, device):
        super().__init__(validate_args=False)
        self.value = torch.tensor([value], dtype=torch.float32, device=device)[0]


    def sample(self, sample_shape=torch.Size()):
        return self.value.view(sample_shape)
    
    def rsample(self, sample_shape: torch.Size = ...) -> torch.Tensor:
        return self.value.view(sample_shape)
    
    def log_prob(self, value):
        return torch.tensor([0], dtype=torch.float32, device=value.device).view_as(value)
    
    @property
    def mean(self):
        return self.value
    
    @property
    def mode(self):
        return self.value
    
    @property
    def variance(self):
        return torch.tensor([0], dtype=torch.float32, device=self.value.device)[0]
    
    def entropy(self):
        return torch.tensor([0], dtype=torch.float32, device=self.value.device)[0]



#safe tanh and atanh from https://github.com/pytorch/rl/blob/main/torchrl/modules/distributions/continuous.py


if version.parse(torch.__version__) >= version.parse("2.0.0"):

    class _SafeTanh(autograd.Function):
        generate_vmap_rule = True

        @staticmethod
        def forward(input, eps):
            output = input.tanh()
            lim = 1.0 - eps
            output = output.clamp(-lim, lim)
            # ctx.save_for_backward(output)
            return output

        @staticmethod
        def setup_context(ctx, inputs, output):
            # input, eps = inputs
            # ctx.mark_non_differentiable(ind, ind_inv)
            # # Tensors must be saved via ctx.save_for_backward. Please do not
            # # assign them directly onto the ctx object.
            ctx.save_for_backward(output)

        @staticmethod
        def backward(ctx, *grad):
            grad = grad[0]
            (output,) = ctx.saved_tensors
            return (grad * (1 - output.pow(2)), None)

    class _SafeaTanh(autograd.Function):
        generate_vmap_rule = True

        @staticmethod
        def setup_context(ctx, inputs, output):
            tanh_val, eps = inputs
            # ctx.mark_non_differentiable(ind, ind_inv)
            # # Tensors must be saved via ctx.save_for_backward. Please do not
            # # assign them directly onto the ctx object.
            ctx.save_for_backward(tanh_val)
            ctx.eps = eps

        @staticmethod
        def forward(tanh_val, eps):
            lim = 1.0 - eps
            output = tanh_val.clamp(-lim, lim)
            # ctx.save_for_backward(output)
            output = output.atanh()
            return output

        @staticmethod
        def backward(ctx, *grad):
            grad = grad[0]
            (tanh_val,) = ctx.saved_tensors
            eps = ctx.eps
            lim = 1.0 - eps
            output = tanh_val.clamp(-lim, lim)
            return (grad / (1 - output.pow(2)), None)

    safetanh = _SafeTanh.apply
    safeatanh = _SafeaTanh.apply

else:

    def safetanh(x, eps):  # noqa: D103
        lim = 1.0 - eps
        y = x.tanh()
        return y.clamp(-lim, lim)

    def safeatanh(y, eps):  # noqa: D103
        lim = 1.0 - eps
        return y.clamp(-lim, lim).atanh()

class SafeTanhTransform(torch.distributions.TanhTransform):
    """TanhTransform subclass that ensured that the transformation is numerically invertible."""

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype.is_floating_point:
            eps = torch.finfo(x.dtype).resolution
        else:
            raise NotImplementedError(f"No tanh transform for {x.dtype} inputs.")
        return safetanh(x, eps)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.dtype.is_floating_point:
            eps = torch.finfo(y.dtype).resolution
        else:
            raise NotImplementedError(f"No inverse tanh for {y.dtype} inputs.")
        x = safeatanh(y, eps)
        return x