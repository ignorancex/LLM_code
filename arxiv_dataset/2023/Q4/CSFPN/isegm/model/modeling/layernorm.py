import torch.nn as nn
import torch.nn.functional as F
import torch


# TODO: instance
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        """
        perform LayerNorm on channel dimension.
        :param num_features: the dimension of the channel.
        :param eps: a value added to the denominator for numerical stability. Default: 1e-5
        :param affine: a boolean value that when set to True, this module has learnable per-element
                    affine parameters initialized to ones (for weights) and zeros (for biases). Default: True.
        """
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
