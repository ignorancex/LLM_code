import torch
import torch.nn as nn

__version__ = '0.0.5'

class APTx(nn.Module):
    r"""The APTx (Alpha Plus Tanh Times) activation function: 
    Research Paper:: APTx: Better Activation Function than MISH, SWISH, and ReLU's Variants used in Deep Learning
    DOI Link: https://doi.org/10.51483/IJAIML.2.2.2022.56-61
    Arxiv: https://arxiv.org/abs/2209.06119
    
    .. math::
        \mathrm{APTx}(x) = (\alpha + \tanh(\beta x)) \cdot \gamma x
    
    :param alpha: Initial α value (default: 1.0)
    :param beta: Initial β value (default: 1.0)
    :param gamma: Initial γ value (default: 0.5)
    :param trainable: If True, all parameters (α, β, γ) become learnable (default: False)
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, trainable=False):
        super().__init__()
        
        # Convert to tensors first
        alpha = torch.as_tensor(float(alpha))
        beta = torch.as_tensor(float(beta))
        gamma = torch.as_tensor(float(gamma))

        if trainable:
            self.alpha = nn.Parameter(alpha)
            self.beta = nn.Parameter(beta)
            self.gamma = nn.Parameter(gamma)
        else:
            self.register_buffer("alpha", alpha)
            self.register_buffer("beta", beta)
            self.register_buffer("gamma", gamma)

    def forward(self, x):
        """Forward pass"""
        return (self.alpha + torch.tanh(self.beta * x)) * self.gamma * x

    def extra_repr(self):
        """Show trainable status in string representation"""
        params = []
        for name in ["alpha", "beta", "gamma"]:
            tensor = getattr(self, name)
            if isinstance(tensor, nn.Parameter):
                params.append(f"{name}=TRAIN")
            else:
                params.append(f"{name}={tensor.item():.2f}")
        return ", ".join(params)
