import argparse
from collections import defaultdict
from functools import partial
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.types

import flash_dropout.functional as F
from benchmark.utils import Benchmarker, make_tensor

torch.backends.cudnn.benchmark = True


def f_cudnn(
    input: torch.Tensor,
    kernel: torch.Tensor,
    d_output: torch.Tensor,
    padding: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
):
    output = torch.nn.functional.conv2d(
        input, kernel, padding=padding, stride=stride, dilation=dilation
    )
    yield "forward"

    output.backward(d_output)
    yield "backward"


def f_im2col(
    input: torch.Tensor,
    kernel: torch.Tensor,
    d_output: torch.Tensor,
    padding: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
):
    # output = F.im2col_conv2d(
    #     input, kernel, padding=padding, stride=stride, dilation=dilation
    # )
    # yield "forward"

    N, C, H, W = input.shape
    K, C, R, S = kernel.shape
    P = (H + 2 * padding[0] - dilation[0] * (R - 1) - 1) // stride[0] + 1
    Q = (W + 2 * padding[1] - dilation[1] * (S - 1) - 1) // stride[1] + 1

    input_im2col = torch.nn.functional.unfold(
        input, kernel_size=(R, S), padding=padding, stride=stride, dilation=dilation
    )  # (N CRS PQ)
    input_im2col = input_im2col.permute(0, 2, 1)  # (N PQ CRS)
    print(input_im2col.shape, input_im2col.stride())

    kernel = kernel.permute(0, 2, 3, 1).view(K, -1)  # (K RSC)

    yield "im2col"

    input_im2col = input_im2col.contiguous()
    yield "input_contiguous"

    # output = torch.nn.functional.linear(input_im2col, kernel)  # (N PQ K)
    output = torch.matmul(input_im2col, kernel.T)
    output = output.permute(0, 2, 1).view(N, K, P, Q)  # (N K PQ)

    yield "forward"

    output.backward(d_output)
    yield "backward"


if __name__ == "__main__":
    import miscellaneous.plot_style

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int)
    parser.add_argument("H", type=int)
    parser.add_argument("W", type=int)
    parser.add_argument("C", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("R", type=int)
    parser.add_argument("S", type=int)
    args = parser.parse_args()

    N, H, W, C, K, R, S = args.N, args.H, args.W, args.C, args.K, args.R, args.S
    padding = (R // 2, S // 2)
    stride = (1, 1)
    dilation = (1, 1)

    P = (H + 2 * padding[0] - dilation[0] * (R - 1) - 1) // stride[0] + 1
    Q = (W + 2 * padding[1] - dilation[1] * (S - 1) - 1) // stride[1] + 1

    L.seed_everything(0)

    input = make_tensor(N, C, H, W)  # .to(memory_format=torch.channels_last)
    kernel = make_tensor(K, C, R, S)  # .to(memory_format=torch.channels_last)
    d_output = make_tensor(N, K, P, Q, requires_grad=False)  # .to(
    #     memory_format=torch.channels_last
    # )

    fns = {}

    fns["CUDNN"] = partial(f_cudnn, input, kernel, d_output, padding, stride, dilation)
    fns["im2col"] = partial(
        f_im2col, input, kernel, d_output, padding, stride, dilation
    )

    benchmarker = Benchmarker(fns, warmup_reps=5, duration=10)
    benchmarker.run()
    df = benchmarker.results(time=np.median)

    df_flops = pd.DataFrame(
        [
            {
                "breakpoint": "forward",
                "FLOPS": 2 * (N * P * Q) * K * (R * S * C),
            },
            {
                "breakpoint": "backward",
                "FLOPS": (
                    2 * (N * H * W) * (R * S * K) * C
                    + 2 * (R * S * C) * (N * P * Q) * K
                ),
            },
        ],
    ).set_index("breakpoint")

    df = df.join(df_flops, on="breakpoint")
    df["TFLOP/s"] = df["FLOPS"] / df["time"] / 1e12
    df.drop(columns=["FLOPS"], inplace=True)

    print(df)
