"""
Code from torchvision.models.densenet modified with cob layers.
https://pytorch.org/docs/stable/torchvision/models.html
"""

import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torch.jit.annotations import List

from neuralteleportation.layers.activation import ReLUCOB
from neuralteleportation.layers.merge import Concat
from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.dropout import DropoutCOB
from neuralteleportation.layers.neuron import BatchNorm2dCOB, LinearCOB, Conv2dCOB
from neuralteleportation.layers.pooling import MaxPool2dCOB, AdaptiveAvgPool2dCOB, AvgPool2dCOB

__all__ = ['DenseNetCOB', 'densenet121COB', 'densenet169COB', 'densenet201COB', 'densenet161COB']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayerCOB(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayerCOB, self).__init__()
        self.add_module('norm1', BatchNorm2dCOB(num_input_features)),
        self.add_module('relu1', ReLUCOB(inplace=True)),
        self.add_module('conv1', Conv2dCOB(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', BatchNorm2dCOB(bn_size * growth_rate)),
        self.add_module('relu2', ReLUCOB(inplace=True)),
        self.add_module('conv2', Conv2dCOB(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient
        self.concat = Concat()
        self.dropout = DropoutCOB(self.drop_rate)

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = self.concat(*inputs)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(*inputs)

        return cp.checkpoint(closure, input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = self.dropout(new_features)
        return new_features


class _DenseBlockCOB(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlockCOB, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayerCOB(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
        self.concat = Concat()

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            if layer is not self.concat:
                new_features = layer(features)
                features.append(new_features)
        return self.concat(*features, dim=1)


class _TransitionCOB(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionCOB, self).__init__()
        self.add_module('norm', BatchNorm2dCOB(num_input_features))
        self.add_module('relu', ReLUCOB(inplace=True))
        self.add_module('conv', Conv2dCOB(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', AvgPool2dCOB(kernel_size=2, stride=2))


class DenseNetCOB(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate, block_config, num_init_features,
                 num_classes, bn_size=4, drop_rate=0, input_channels=3,
                 memory_efficient=False):

        super(DenseNetCOB, self).__init__()                     
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', Conv2dCOB(input_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', BatchNorm2dCOB(num_init_features)),
            ('relu0', ReLUCOB(inplace=True)),
            ('pool0', MaxPool2dCOB(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlockCOB(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _TransitionCOB(num_input_features=num_features,
                                       num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', BatchNorm2dCOB(num_features))

        # Flatten layer
        self.relu = ReLUCOB(inplace=True)
        self.adaptive_avg_pool2d = AdaptiveAvgPool2dCOB((1, 1))
        self.flatten = FlattenCOB()

        # Linear layer
        self.classifier = LinearCOB(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, Conv2dCOB):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, BatchNorm2dCOB):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, LinearCOB):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.relu(features)
        out = self.adaptive_avg_pool2d(out)
        out = self.flatten(out)
        out = self.classifier(out)
        return out


def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenetCOB(arch, growth_rate, block_config, num_init_features, pretrained, progress,
                 **kwargs):
    model = DenseNetCOB(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121COB(pretrained=False, progress=True, **kwargs):
    """Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenetCOB('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                        **kwargs)


def densenet161COB(pretrained=False, progress=True, **kwargs):
    """Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenetCOB('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                        **kwargs)


def densenet169COB(pretrained=False, progress=True, **kwargs):
    """Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenetCOB('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                        **kwargs)


def densenet201COB(pretrained=False, progress=True, **kwargs):
    """Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenetCOB('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                        **kwargs)


if __name__ == '__main__':
    from tests.model_test import test_teleport

    densenet = densenet121COB(pretrained=True, num_classes=1000)
    test_teleport(densenet, (1, 3, 224, 224), verbose=True)
