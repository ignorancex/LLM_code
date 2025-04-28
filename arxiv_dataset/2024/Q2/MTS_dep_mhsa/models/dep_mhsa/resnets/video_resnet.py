import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

# test attention mechanism
from models.dep_mhsa.attentions.mhsa_collection import MHSA3D, DepMHSA
from models.dep_mhsa.attentions.single_module_attn import SingleModalAtten
from models.dep_mhsa.attentions.channel_attention import CAB3D
from models.dep_mhsa.resnets.blocks import BasicBlock, BasicBlockWithAttention, Bottleneck
from models.dep_mhsa.resnets.stems import R2Plus1dStem, BasicStem


class Conv3DNoTemporal(nn.Conv3d):
    def __init__(
        self, in_planes: int, out_planes: int, midplanes: Optional[int] = None, stride: int = 1, padding: int = 1
    ) -> None:

        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return 1, stride, stride
    
class Conv3DSimple(nn.Conv3d):
    def __init__(
        self, in_planes: int, out_planes: int, midplanes: Optional[int] = None, stride: int = 1, padding: int = 1
    ) -> None:

        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return stride, stride, stride


class Conv2DPlus1D(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, midplanes: int, stride: int = 1, padding:int = 1) -> None:
        super().__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes, out_planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False
            ),
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return stride, stride, stride

class VideoResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck, BasicBlockWithAttention]],
        conv_makers: Sequence[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2DPlus1D]]],
        layers: List[int],
        stem: Callable[..., nn.Module],
        num_classes: int = 400,
        zero_init_residual: bool = False,
        **kwargs
    ) -> None:
        """Generic resnet video generator.

        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): resnet building block
            conv_makers (List[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]]): generator
                function for each layer
            layers (List[int]): number of blocks per layer
            stem (Callable[..., nn.Module]): module specifying the ResNet stem.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super().__init__()
        self.inplanes = 64
        self.n_frames_list = kwargs['n_frames_list']
        self.n_resolution_list = kwargs['input_last_layers']
        attn_makers= kwargs['attn_makers']
        self.stem = stem()
        
        self.layer1 = self._make_layer(block, conv_makers[0], attn_makers[0], 64, 
                                        layers[0], stride=1, resolution_last_layer=self.n_resolution_list[0], 
                                        n_frames_last_layer=self.n_frames_list[0], **kwargs)


        self.layer2 = self._make_layer(block, conv_makers[1], attn_makers[1], 128, 
                                        layers[1], stride=2, resolution_last_layer=self.n_resolution_list[1],
                                        n_frames_last_layer=self.n_frames_list[1], **kwargs)


        self.layer3 = self._make_layer(block, conv_makers[2], attn_makers[2], 256, 
                                        layers[2], stride=2, resolution_last_layer=self.n_resolution_list[2],
                                        n_frames_last_layer=self.n_frames_list[2],**kwargs)


        self.layer4 = self._make_layer(block, conv_makers[3], attn_makers[3], 512, 
                                        layers[3], stride=2, resolution_last_layer=self.n_resolution_list[3],
                                        n_frames_last_layer=self.n_frames_list[3],**kwargs)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
                                

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[union-attr, arg-type]

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x) # output: torch.Size([64, 64, 11, 64, 64])
        # BatchSize, Channel, Frames, H, W
        x = self.layer1(x) # output: torch.Size([BatchSize, 64, 11, 64, 64])    
        x = self.layer2(x) # output: torch.Size([BatchSize, 128, 6, 32, 32])
        x = self.layer3(x) # output" torch.Size([BatchSize, 256, 3, 16, 16])
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        conv_builder: Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2DPlus1D]],
        attn_builder: Type[Union[MHSA3D, SingleModalAtten, DepMHSA, CAB3D]],
        planes: int,
        blocks: int,
        stride: int = 1,
        **kwargs
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, attn_builder, stride, downsample, 
                            **kwargs))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder, attn_builder, 
                                # resolution_last_layer=resolution_last_layer, 
                                # n_frames_last_layer=n_frames_last_layer, 
                                **kwargs))

        return nn.Sequential(*layers)


def _video_resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    conv_makers: Sequence[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2DPlus1D]]],
    layers: List[int],
    stem: Callable[..., nn.Module],
    **kwargs: Any,
) -> VideoResNet:
    model = VideoResNet(block, conv_makers, layers, stem, **kwargs)
    return model