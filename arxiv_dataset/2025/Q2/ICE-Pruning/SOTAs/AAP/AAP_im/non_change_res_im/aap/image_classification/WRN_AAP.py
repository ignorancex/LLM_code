import math
from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.nn.init as init
from itertools import repeat
from torch.nn import Parameter
import pickle

outputs = {}
def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def _ntuple(n):
    def parse(x):
        if isinstance(x, tuple):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)

class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, layer='', first=True, second=False):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.layer = layer
        self.first = first
        self.second = second

        #self.outputs = {}

        self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels // self.groups, *self.kernel_size))
        self.mask = Parameter(torch.ones([self.out_channels, self.in_channels // self.groups, *self.kernel_size]), requires_grad=False)

        if first:
            self.bn1 = nn.BatchNorm2d(self.out_channels)
        elif second:
            self.bn2 = nn.BatchNorm2d(self.out_channels)
        else:
            self.bn3 = nn.BatchNorm2d(self.out_channels)

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels=' + str(self.in_channels) \
            + ', out_channels=' + str(self.out_channels) \
            + ', kernel_size=' + str(self.kernel_size) \
            + ', bias=' + str(self.bias is not None) + ')'

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = F.conv2d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        if self.first:
            block = 'conv1'
            output = self.bn1(output)
        elif self.second:
            block = 'conv2'
            output = self.bn2(output)
        else:
            block = 'conv3'
            output = self.bn3(output)

        tmp_output = F.relu(output)
        
        name = str(self.weight.data.device) + '.' + self.layer + '.' + block
        
        outputs[name] = tmp_output
        
        return output

    def prune(self, threshold, resample, reinit=False):
        # print("conv, prune weights, percentile_value: {}".format(threshold))
        weight_dev = self.weight.device
        mask_dev = self.mask.device

        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)

        # Apply new weight and mask
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

        if reinit:
            nn.init.xavier_uniform_(self.weight)
            self.weight.data = self.weight.data * self.mask.data

    def prune_filters(self, prune_index, resample, reinit=False):
        # print("conv, prune {} filters".format(len(prune_index)))
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        weight_arr = self.weight.data.cpu().numpy()
        new_mask = self.mask.data.cpu().numpy()
        new_mask[prune_index, :, :, :] = 0
        self.weight.data = torch.from_numpy(weight_arr * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

    def mask_on_input_channels(self, prune_index, resample, reinit=False):
        #print("mask {} input channels".format(len(prune_index)))
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        weight_arr = self.weight.data.cpu().numpy()
        new_mask = self.mask.data.cpu().numpy()
        new_mask[:, prune_index, :, :] = 0
        self.weight.data = torch.from_numpy(weight_arr * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, layer: str='', first: bool=True, second: bool=False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    '''
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
    '''
    return MaskedConv2d(in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        layer=layer,
        first=first,
        second=second)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, layer: str='', first: bool=True, second: bool=False) -> nn.Conv2d:
    """1x1 convolution"""
    #return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return MaskedConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, layer=layer, first=first, second=second)

def conv1x1_ori(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        layer: str = '',
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, layer=layer)
        #self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, layer=layer, first=False, second=True)
        #self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, layer=layer, first=False, second=False)
        #self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        #out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64 * 2,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], layer='layer1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], layer='layer2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], layer='layer3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], layer='layer4')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.outputs = {}

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        layer: str = '',
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_ori(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, layer+'.0'
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    layer=layer + '.' + str(i)
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        #print(x.shape[0])
        if x.shape[0] == 40 and not self.training:
            print('saving the outputs')
            with open('outputs.pkl', 'wb') as handle:
                pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
) -> ResNet:
    model = ResNet(block, layers)

    return model

def wide_resnet101_2() -> ResNet:
    return _resnet(Bottleneck, [3, 4, 23, 3])
