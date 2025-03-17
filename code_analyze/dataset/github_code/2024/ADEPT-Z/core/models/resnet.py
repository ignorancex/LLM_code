"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:24:50
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:36:15
"""

from typing import Union

import torch
import torch.nn.functional as F
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn
from torch.nn.modules.activation import ReLU
from torch.types import _size

from core.cost.ADC import ADC_list
from core.cost.DAC import DAC_list
from core.models.layers.activation import ReLUN

__all__ = [
    "ResNet18",
    "ResNet20",
    "ResNet32",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
]


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation: bool = True,
        act_thres: float = 6.0,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = ReLUN(act_thres, inplace=True) if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def conv3x3(
    in_planes,
    out_planes,
    bias: bool = False,
    stride: Union[int, _size] = 1,
    padding: Union[int, _size] = 0,
):
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
    )

    return conv


def conv1x1(
    in_planes,
    out_planes,
    bias: bool = False,
    stride: Union[int, _size] = 1,
    padding: Union[int, _size] = 0,
):
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=padding,
        bias=bias,
    )

    return conv


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        # unique parameters
        act_thres: int = 6,
    ) -> None:
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(
        #     in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = conv3x3(
            in_planes,
            planes,
            bias=False,
            stride=stride,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        )
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        self.conv2 = conv3x3(
            planes,
            planes,
            bias=False,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        )

        self.shortcut = nn.Identity()
        # self.shortcut.conv1_spatial_sparsity = self.conv1.bp_input_sampler.spatial_sparsity
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(
                    in_planes,
                    self.expansion * planes,
                    bias=False,
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        # unique parameters
        act_thres: int = 6,
    ) -> None:
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv1 = conv1x1(
            in_planes,
            planes,
            bias=False,
            stride=1,
            padding=0,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        )
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = conv3x3(
            planes,
            planes,
            bias=False,
            stride=stride,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        )
        # self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.conv3 = conv1x1(
            planes,
            self.expansion * planes,
            bias=False,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.act3 = (
            ReLUN(act_thres, inplace=True) if act_thres <= 6 else ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                conv1x1(
                    in_planes,
                    self.expansion * planes,
                    bias=False,
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act3(out)
        return out


class ResNet(nn.Module):
    """ResNet"""

    _conv_linear = (
        nn.Conv2d,
        nn.Linear,
    )

    def __init__(
        self,
        block,
        num_blocks,
        in_planes,
        img_height,
        img_width,
        in_channels,
        num_classes,
        bias=False,
        act_thres=6,
        device="cuda:0",
        bn_affine=True,
    ) -> None:
        # resnet params
        self.block = block
        self.num_blocks = num_blocks
        self.in_planes = in_planes
        self.img_height = img_height
        self.img_width = img_width
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bias = bias
        self.act_thres = act_thres
        self.bn_affine = bn_affine
        self.device = device
        super().__init__()
        self.build_layers()

    def build_layers(self):
        # build layers
        block = self.block
        in_planes = self.in_planes
        num_blocks = self.num_blocks
        # build layers
        blkIdx = 0
        self.conv1 = conv3x3(
            self.in_channels,
            in_planes,
            bias=False,
            stride=1
            if self.img_height <= 64
            else 2,  # downsample for imagenet, dogs, cars
            padding=1,
        )
        # self.conv1 = nn.Conv2d(in_channels, in_planes, 3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(in_planes)
        blkIdx += 1

        self.layer1 = self._make_layer(
            block,
            in_planes,
            num_blocks[0],
            stride=1,
            act_thres=self.act_thres,
        )
        blkIdx += 1

        self.layer2 = self._make_layer(
            block,
            in_planes * 2,
            num_blocks[1],
            stride=2,
            act_thres=self.act_thres,
        )
        blkIdx += 1

        self.layer3 = self._make_layer(
            block,
            in_planes * 4,
            num_blocks[2],
            stride=2,
            act_thres=self.act_thres,
        )
        blkIdx += 1

        self.layer4 = self._make_layer(
            block,
            in_planes * 8,
            num_blocks[3],
            stride=2,
            act_thres=self.act_thres,
        )
        blkIdx += 1

        n_channel = in_planes * 8 if num_blocks[3] > 0 else in_planes * 4

        self.linear = LinearBlock(
            n_channel * block.expansion,
            self.num_classes,
            bias=False,
            activation=False,
            act_thres=self.act_thres,
        )

        self.drop_masks = None

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride,
        # unique parameters
        **kwargs,
    ):
        if num_blocks == 0:
            return nn.Identity()
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    **kwargs,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def reset_parameters(self, random_state=None):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def get_parameter_groups(self, weight_decay: float = 0, lr: float = 1e-3):
        param_optimizer = list(self.named_parameters())

        no_decay = ["bias", "weight"]
        no_decay_group = [
            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
        ]
        no_decay_reduce_lr_group = []
        for m in self.modules():
            if isinstance(m, self._conv_linear):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        no_decay_reduce_lr_group.append(p)
        no_decay_group = list(set(no_decay_group) - set(no_decay_reduce_lr_group))
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": lr,
            },  # no decay for finetuning/distill
            {"params": no_decay_group, "weight_decay": 0.0, "lr": lr},
            {"params": no_decay_reduce_lr_group, "weight_decay": 0.0, "lr": lr / 10},
        ]
        return optimizer_grouped_parameters

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if x.size(-1) > 64:  # 224 x 224, e.g., cars, dogs, imagenet
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)

        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def ResNet18(*args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], 64, *args, **kwargs)


def ResNet20(*args, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3, 0], 16, *args, **kwargs)


def ResNet32(*args, **kwargs):
    return ResNet(BasicBlock, [5, 5, 5, 0], 16, *args, **kwargs)


def ResNet34(*args, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], 64, *args, **kwargs)


def ResNet50(*args, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], 64, *args, **kwargs)


def ResNet101(*args, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], 64, *args, **kwargs)


def ResNet152(*args, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], 64, *args, **kwargs)


def test():
    device = torch.device("cuda")

    ps_cost = {
        "width": 85,
        "height": 80,
        "static_power": 14.8,
        "dynamic_power": 10,
        "insertion_loss": 0.1,
    }
    dc_cost = {"width": 50, "length": 30, "insertion_loss": 0.3}
    dc2_cost = {"width": 50, "length": 30, "insertion_loss": 0.3}
    dc3_cost = {"width": 75, "length": 45, "insertion_loss": 0.3}
    dc4_cost = {"width": 100, "length": 60, "insertion_loss": 0.3}
    dc5_cost = {"width": 125, "length": 75, "insertion_loss": 0.3}
    dc6_cost = {"width": 150, "length": 90, "insertion_loss": 0.3}
    dc7_cost = {"width": 175, "length": 105, "insertion_loss": 0.3}
    dc8_cost = {"width": 200, "length": 120, "insertion_loss": 0.3}
    cr_cost = {"width": 8, "height": 8, "cr_spacing": 10, "insertion_loss": 0.1}
    # photodetector_cost = {'sensitivity': -5, 'power': 2.8, 'width': 40, 'length': 40, 'latency':10}
    photodetector_cost = {
        "sensitivity": -25,
        "power": 2.8,
        "width": 40,
        "length": 40,
        "latency": 10,
    }
    TIA_cost = {"power": 3, "area": 5200, "latency": 10}
    modulator_cost = {
        "static_power": 10,
        "width": 50,
        "length": 300,
        "insertion_loss": 0.8,
    }
    attenuator_cost = {
        "insertion_loss": 0.1,
        "length": 7.5,
        "width": 7.5,
        "static_power": 2.5,
        "dynamic_power": 0,
    }

    k = 16

    arch = arch = dict(
        n_waveguides=k,
        n_front_share_waveguides=k,
        n_front_share_ops=k,
        n_blocks=k,
        n_layers_per_block=2,
        n_front_share_blocks=4,
        share_ps="none",
        interleave_dc=True,
        device_cost=dict(
            ps_cost=ps_cost,
            dc_cost=dc_cost,
            dc2_cost=dc2_cost,
            dc3_cost=dc3_cost,
            dc4_cost=dc4_cost,
            dc5_cost=dc5_cost,
            dc6_cost=dc6_cost,
            dc7_cost=dc7_cost,
            dc8_cost=dc8_cost,
            cr_cost=cr_cost,
            photodetector_cost=photodetector_cost,
            TIA_cost=TIA_cost,
            modulator_cost=modulator_cost,
            attenuator_cost=attenuator_cost,
            adc_cost=ADC_list,
            dac_cost=DAC_list,
            laser_wall_plug_eff=0.25,
            spacing=50,  # unit um
            area_upper_bound=100000,
            area_lower_bound=100,
            first_active_block=True,
            resolution=4,
            n_group=4.5,  # Group index
        ),
        dc_port_candidates=[2, 3, 4, 6, 8],
    )

    net = ResNet20(
        # in_channel3=3,
        img_height=32,
        img_width=32,
        in_channels=3,
        num_classes=10,
        # block_list=[8, 8, 8, 8, 8, 8],
        block_list=[16, 16, 16, 16, 16, 16],
        in_bit=32,
        w_bit=32,
        v_max=10.8,
        # v_pi=4.36,
        act_thres=6,
        photodetect=True,
        super_layer_name="ps_dc_cr_adeptzero",
        super_layer_config=arch,
        device=device,
    ).to(device)

    x = torch.randn(2, 3, 32, 32).to(device)
    # print(net)
    y = net(x)
    print(y.shape)


if __name__ == "__main__":
    test()
