import torch
from torch import nn

from core.models.layers.super_conv2d import SuperBlockConv2d
from core.models.layers.super_linear import SuperBlockLinear

__all__ = ["network_weight_gaussian_init"]


def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, SuperBlockConv2d):
                # nn.init.normal_(m.weight)
                # # gaussian initialize weight matrix
                m.reset_parameters()  # svd-based method to decompose a gaussian matrix
                # if hasattr(m, "bias") and m.bias is not None:
                #     nn.init.zeros_(m.bias)
                for layer in m.super_ps_layers:  # random initialize phases
                    layer.reset_parameters("uniform")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, SuperBlockLinear):
                # nn.init.normal_(m.weight)
                m.reset_parameters()
                # if hasattr(m, "bias") and m.bias is not None:
                #     nn.init.zeros_(m.bias)
                for layer in m.super_ps_layers:  # random initialize phases
                    layer.reset_parameters("uniform")
            else:
                continue

    return net
