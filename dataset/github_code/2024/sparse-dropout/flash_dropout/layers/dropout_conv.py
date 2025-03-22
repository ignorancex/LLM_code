import math

import torch
import torch.types
from torch import nn

import flash_dropout.functional as F


class DropoutConv2d(nn.Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "in_channels",
        "out_channels",
        "kernel_size",
        "p",
        "variant",
    ]
    p: float
    variant: str

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        p: float = 0.1,
        variant: str = "vanilla",
        **kwargs,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.p = p
        self.variant = variant
        self.kwargs = kwargs

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        def standard_conv(input: torch.Tensor) -> torch.Tensor:
            return nn.functional.conv2d(
                input,
                self.weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

        if not self.training:
            return standard_conv(input)

        match self.variant:
            case "none":
                output = standard_conv(input)
            case "im2col":
                output = F.im2col_conv2d(
                    input,
                    self.weight,
                    padding=self.padding,
                    stride=self.stride,
                    dilation=self.dilation,
                )
            case "vanilla":
                input = nn.functional.dropout(input, p=self.p)
                output = standard_conv(input)
            case "vanilla_2d":
                input = nn.functional.dropout2d(input, p=self.p)
                output = standard_conv(input)
            # case "blockwise[naive]":
            #     output = F.naive_blockwise_dropout_matmul(
            #         input, self.weight, self.kwargs["block_size"], self.p
            #     )
            # case "blockwise[triton]":
            #     output = F.triton_blockwise_dropout_matmul(
            #         input, self.weight, self.kwargs["block_size"], self.p
            #     )
            # case "blockwise[cuda]":
            #     output = F.cuda_blockwise_dropout_matmul(
            #         input, self.weight, self.kwargs["block_size"], self.p
            #     )
            # case _:
            #     raise ValueError(f"Unknown variant {self.variant}")

        return output
