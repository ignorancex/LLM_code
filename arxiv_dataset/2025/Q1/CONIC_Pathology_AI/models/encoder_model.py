import os
from collections import OrderedDict

import torch
import torch.nn as nn

from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet
from backbones.encoders import get_encoder
from timm.models.senet import SENet, SEResNetBottleneck, SEResNeXtBottleneck
# from timm.models.convnext import ConvNeXt, ConvNeXtBlock, ConvNeXtStage

class ResNetExt(ResNet):
    def _forward_impl(self, x, freeze):
        # See note [TorchScript super()]
        if self.training:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            with torch.set_grad_enabled(not freeze):
                x1 = x = self.layer1(x)
                x2 = x = self.layer2(x)
                x3 = x = self.layer3(x)
                x4 = x = self.layer4(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x1 = x = self.layer1(x)
            x2 = x = self.layer2(x)
            x3 = x = self.layer3(x)
            x4 = x = self.layer4(x)
        return x1, x2, x3, x4

    def forward(self, x: torch.Tensor, freeze: bool = False) -> torch.Tensor:
        return self._forward_impl(x, freeze)

    @staticmethod
    def resnet50(num_input_channels, pretrained=None):
        model = ResNetExt(ResNetBottleneck, [3, 4, 6, 3])
        model.conv1 = nn.Conv2d(
                num_input_channels, 64, 7, stride=1, padding=3)
        
        if pretrained is None:
            return model
            
        if pretrained is not None and os.path.exists(pretrained):
            print(f"Loading: {pretrained}")
            pretrained = torch.load(pretrained)
            (
                    missing_keys, unexpected_keys
            ) = model.load_state_dict(pretrained, strict=False)
            print("missing keys: ", missing_keys)
            print("unexpected keys: ", unexpected_keys)
        elif not os.path.exists(pretrained):
            assert os.path.exists(pretrained), \
                    f"Pretrained path is not valid: {pretrained}"
        return model

    @staticmethod
    def resnet101(num_input_channels, pretrained=None):
        model = ResNetExt(ResNetBottleneck, [3, 4, 23, 3])
        model.conv1 = nn.Conv2d(
                num_input_channels, 64, 7, stride=1, padding=3)
        
        if pretrained is None:
            return model
            
        if pretrained is not None and os.path.exists(pretrained):
            print(f"Loading: {pretrained}")
            pretrained = torch.load(pretrained)
            (
                    missing_keys, unexpected_keys
            ) = model.load_state_dict(pretrained, strict=False)
            print("missing keys: ", missing_keys)
            print("unexpected keys: ", unexpected_keys)
        elif not os.path.exists(pretrained):
            assert os.path.exists(pretrained), \
                    f"Pretrained path is not valid: {pretrained}"
        return model
    
    @staticmethod
    def resnext101(num_input_channels, pretrained=None):
        model = ResNetExt(ResNetBottleneck, [3, 4, 23, 3], groups=32, width_per_group=8)
        model.conv1 = nn.Conv2d(
                num_input_channels, 64, 7, stride=1, padding=3)
        
        print("model.groups: ", model.groups)
        print("model.base_width: ", model.base_width)
        if pretrained is None:
            return model

        if pretrained is not None and os.path.exists(pretrained):
            print(f"Loading: {pretrained}")
            pretrained = torch.load(pretrained)
            (
                    missing_keys, unexpected_keys
            ) = model.load_state_dict(pretrained, strict=False)
            print("missing keys: ", missing_keys)
            print("unexpected keys: ", unexpected_keys)
        elif not os.path.exists(pretrained):
            assert os.path.exists(pretrained), \
                    f"Pretrained path is not valid: {pretrained}"
        return model
    
    @staticmethod
    def resnext50(num_input_channels, pretrained=None):
        model = ResNetExt(ResNetBottleneck, [3, 4, 6, 3], groups=32, width_per_group=4)
        model.conv1 = nn.Conv2d(
                num_input_channels, 64, 7, stride=1, padding=3)
        
        print("model.groups: ", model.groups)
        print("model.base_width: ", model.base_width)
        if pretrained is None:
            return model
        if pretrained is not None and os.path.exists(pretrained):
            print(f"Loading: {pretrained}")
            pretrained = torch.load(pretrained)
            (
                    missing_keys, unexpected_keys
            ) = model.load_state_dict(pretrained, strict=False)
            print("missing keys: ", missing_keys)
            print("unexpected keys: ", unexpected_keys)
        elif not os.path.exists(pretrained):
            assert os.path.exists(pretrained), \
                    f"Pretrained path is not valid: {pretrained}"
        return model

class SeResNextExt(SENet):
    def _forward_impl(self, x, freeze):
        # See note [TorchScript super()]
        if self.training:
            x = self.layer0(x)
            with torch.set_grad_enabled(not freeze):
                x1 = x = self.layer1(x)
                x2 = x = self.layer2(x)
                x3 = x = self.layer3(x)
                x4 = x = self.layer4(x)
        else:
            x = self.layer0(x)
            x1 = x = self.layer1(x)
            x2 = x = self.layer2(x)
            x3 = x = self.layer3(x)
            x4 = x = self.layer4(x)
        return x1, x2, x3, x4

    def forward(self, x: torch.Tensor, freeze: bool = False) -> torch.Tensor:
        return self._forward_impl(x, freeze)

    @staticmethod
    def seresnet50(num_input_channels, pretrained=None):
        model = SeResNextExt(SEResNeXtBottleneck, [3, 4, 6, 3], groups=1, reduction=16)
        layer0_modules = [
                ('conv1', nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        model.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        if pretrained is None:
            return model
            
        if pretrained is not None and os.path.exists(pretrained):
            print(f"Loading: {pretrained}")
            pretrained = torch.load(pretrained)
            (missing_keys, unexpected_keys) = model.load_state_dict(pretrained, strict=False)
            print("missing keys: ", missing_keys)
            print("unexpected keys: ", unexpected_keys)
        elif not os.path.exists(pretrained):
            assert os.path.exists(pretrained), \
                    f"Pretrained path is not valid: {pretrained}"
        return model


    @staticmethod
    def seresnext50_32x4d(num_input_channels, pretrained=None):
        print("loading seresnext 50 32 x4d")
        print("==================================================================")

        model = SeResNextExt(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16)
        layer0_modules = [
                ('conv1', nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        model.layer0 = nn.Sequential(OrderedDict(layer0_modules))
         
        if pretrained is None:
            return model
            
        if pretrained is not None and os.path.exists(pretrained):
            print(f"Loading: {pretrained}")
            pretrained = torch.load(pretrained)
            (missing_keys, unexpected_keys) = model.load_state_dict(pretrained, strict=False)
            print("missing keys: ", missing_keys)
            print("unexpected keys: ", unexpected_keys)
        elif not os.path.exists(pretrained):
            assert os.path.exists(pretrained), \
                    f"Pretrained path is not valid: {pretrained}"
        return model
    
    @staticmethod
    def seresnext50_32x4d_aug(num_input_channels, pretrained=None):
        print("loading seresnext 50 32 x4d")
        print("==================================================================")

        model = SeResNextExt(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16)
            
        if pretrained is not None and os.path.exists(pretrained):
            print(f"Loading: {pretrained}")
            pretrained = torch.load(pretrained)
            (missing_keys, unexpected_keys) = model.load_state_dict(pretrained, strict=False)
            print("missing keys: ", missing_keys)
            print("unexpected keys: ", unexpected_keys)
        
        layer0_modules = [
                ('conv1', nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        model.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        return model

    @staticmethod
    def seresnext101_32x4d(num_input_channels, pretrained=None):
        print("loading seresnext101_32x4d")
        print("==================================================================")

        model = SeResNextExt(SEResNeXtBottleneck, layers=[3, 4, 23, 3], groups=32, reduction=16)
        layer0_modules = [
                ('conv1', nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        model.layer0 = nn.Sequential(OrderedDict(layer0_modules))
         
        if pretrained is None:
            return model
            
        if pretrained is not None and os.path.exists(pretrained):
            print(f"Loading: {pretrained}")
            pretrained = torch.load(pretrained)
            (missing_keys, unexpected_keys) = model.load_state_dict(pretrained, strict=False)
            print("missing keys: ", missing_keys)
            print("unexpected keys: ", unexpected_keys)
        elif not os.path.exists(pretrained):
            assert os.path.exists(pretrained), \
                    f"Pretrained path is not valid: {pretrained}"
        return model

"""
class ConvNeXtExt(ConvNeXt):
    def _forward_impl(self, x, freeze):
        # See note [TorchScript super()]
        if self.training:
            x = self.layer0(x)
            with torch.set_grad_enabled(not freeze):
                x1 = x = self.layer1(x)
                x2 = x = self.layer2(x)
                x3 = x = self.layer3(x)
                x4 = x = self.layer4(x)
        else:
            x = self.layer0(x)
            x1 = x = self.layer1(x)
            x2 = x = self.layer2(x)
            x3 = x = self.layer3(x)
            x4 = x = self.layer4(x)
        return x1, x2, x3, x4

    def forward(self, x: torch.Tensor, freeze: bool = False) -> torch.Tensor:
        return self._forward_impl(x, freeze)

    @staticmethod
    def seresnet50(num_input_channels, pretrained=None):
        model = SeResNextExt(SEResNeXtBottleneck, [3, 4, 6, 3], groups=1, reduction=16)
        layer0_modules = [
                ('conv1', nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        model.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        if pretrained is None:
            return model
            
        if pretrained is not None and os.path.exists(pretrained):
            print(f"Loading: {pretrained}")
            pretrained = torch.load(pretrained)
            (missing_keys, unexpected_keys) = model.load_state_dict(pretrained, strict=False)
            print("missing keys: ", missing_keys)
            print("unexpected keys: ", unexpected_keys)
        elif not os.path.exists(pretrained):
            assert os.path.exists(pretrained), \
                    f"Pretrained path is not valid: {pretrained}"
        return model
"""