"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-04-01 19:43:53
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-09-27 21:42:08
"""

import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import shapely.ops as so
import torch
from pyutils.general import ensure_dir, logger
from pyutils.quantize import uniform_quantize
from pyutils.torch_train import set_torch_deterministic
from shapely.geometry import Polygon
from torch import Tensor
from torch.types import _size

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))


__all__ = [
    "sinkhorn",
    "assert_unitary",
    "GradientMask",
    "weight_quantize_fn",
    "clip_grad_value_",
    "diff_round",
    "hard_diff_round",
    "PermutationFunction",
]


def sinkhorn(w, n_step=20, t_min=0.1, noise_std=0.01, svd=False):
    with torch.no_grad():
        if svd:
            u, _, v = w.svd()
            w = u.matmul(v.permute(-1, -2))
        if noise_std > 0:
            w = (
                w
                + torch.eye(w.size(-1), device=w.device).mul_(noise_std)
                + torch.randn_like(w).mul_(noise_std)
            )
        w = w.abs()
        for step in range(n_step):
            t = t_min ** (step / n_step)
            # t = t_min
            w /= w.sum(dim=-2, keepdim=True)
            w /= w.sum(dim=-1, keepdim=True)
            w = w.div(t).softmax(dim=-1)
        return w


class GradientMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, idx: Optional[int], grad_ratio: float):
        ctx.idx = idx
        ctx.grad_ratio = grad_ratio
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        idx = ctx.idx
        if idx is None:  # penalize all
            grad_input = grad_output.clone()
        elif idx == -1:  # no penalty
            grad_input = torch.zeros_like(grad_output)
        else:  # penalize selected
            grad_input = torch.zeros_like(grad_output)
            grad_input[idx] = grad_output[idx] * ctx.grad_ratio
        return grad_input, None, None


def clip_grad_value_(parameters, clip_value: float):
    for p in parameters:
        if p.grad is not None:
            if p.grad.is_complex():
                p.grad.data.real.clamp_(min=-clip_value, max=clip_value)
                p.grad.data.imag.clamp_(min=-clip_value, max=clip_value)
            else:
                p.grad.data.clamp_(min=-clip_value, max=clip_value)


class RoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        mask = (x.max(dim=1, keepdim=True)[0] > 0.95).repeat(1, x.size(-1))
        return torch.where(mask, x.round(), x)

    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output.clone()


class HardRoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        # x_max, indices = x.max(dim=1, keepdim=True)
        # illegal_indices = [k for k, v in Counter(indices.view(-1).cpu().numpy().tolist()).items() if v > 1]
        # mask = x_max > 0.95
        # for i in illegal_indices:

        mask = (x.max(dim=1, keepdim=True)[0] > 0.9).repeat(1, x.size(-1))
        ctx.mask = mask
        return torch.where(mask, x.round(), x)

    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output.clone().masked_fill_(ctx.mask, 0)


def diff_round(x: Tensor) -> Tensor:
    """Project to closest permutation matrix"""
    assert x.size(-1) == x.size(
        -2
    ), f"input x has to be a square matrix, but got {x.size()}"
    return RoundFunction.apply(x)


def hard_diff_round(x: Tensor) -> Tensor:
    """Project to closest permutation matrix"""
    assert x.size(-1) == x.size(
        -2
    ), f"input x has to be a square matrix, but got {x.size()}"
    return HardRoundFunction.apply(x)


class weight_quantize_fn(torch.nn.Module):
    def __init__(self, w_bit, mode="oconv", alg="dorefa", quant_ratio=1.0):
        """Differentiable weight quantizer. Support different algorithms. Support Quant-Noise with partial quantization.

        Args:
            w_bit (int): quantization bitwidth
            mode (str, optional): Different mode indicates different NN architectures. Defaults to "oconv".
            alg (str, optional): Quantization algorithms. [dorefa, dorefa_sym, qnn, dorefa_pos] Defaults to "dorefa".
            quant_ratio (float, optional): Quantization ratio to support full-precision gradient flow. Defaults to 1.0.
        """
        super(weight_quantize_fn, self).__init__()
        assert 1 <= w_bit <= 32, logger.error(
            f"Only support 1 - 32 bit quantization, but got {w_bit}"
        )
        self.w_bit = w_bit
        self.alg = alg
        self.mode = mode
        assert alg in {"dorefa", "dorefa_sym", "qnn", "dorefa_pos"}, logger.error(
            f"Only support (dorefa, dorefa_sym, qnn, dorefa_pos) algorithms, but got {alg}"
        )
        self.quant_ratio = quant_ratio
        assert 0 <= quant_ratio <= 1, logger.error(
            f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}"
        )
        self.uniform_q = uniform_quantize(k=w_bit, gradient_clip=True)

    def set_quant_ratio(self, quant_ratio=None):
        if quant_ratio is None:
            ### get recommended value
            quant_ratio = [
                None,
                0.2,
                0.3,
                0.4,
                0.5,
                0.55,
                0.6,
                0.7,
                0.8,
                0.83,
                0.86,
                0.89,
                0.92,
                0.95,
                0.98,
                0.99,
                1,
            ][min(self.w_bit, 16)]
        assert 0 <= quant_ratio <= 1, logger.error(
            f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}"
        )
        self.quant_ratio = quant_ratio

    def forward(self, x):
        if self.quant_ratio < 1 and self.training:
            ### implementation from fairseq
            ### must fully quantize during inference
            quant_noise_mask = torch.empty_like(x, dtype=torch.bool).bernoulli_(
                1 - self.quant_ratio
            )
        else:
            quant_noise_mask = None

        if self.w_bit == 32:
            weight_q = torch.tanh(x)
            weight_q = weight_q / torch.max(torch.abs(weight_q))
        elif self.w_bit == 1:
            if self.mode == "ringonn":
                weight_q = (self.uniform_q(x) / 4) + 0.5
            else:
                if self.alg == "dorefa":
                    weight_q = (
                        self.uniform_q(x)
                        .add(1)
                        .mul((1 - 2**0.5 / 2) / 2)
                        .add(2**0.5 / 2)
                    )  # [0.717, 1]
                    if quant_noise_mask is not None:
                        x = x.add((2 + 2**0.5) / 4)  # mean is (0.717+1)/2
                        noise = weight_q.data.sub_(x.data).masked_fill_(
                            quant_noise_mask, 0
                        )
                        ### unquantized weights have to follow reparameterization, i.e., tanh and scale
                        weight_q = x + noise
                elif self.alg == "dorefa_sym":
                    E = x.data.abs().mean()
                    weight_q = self.uniform_q(x / E) * E  # [-E, E]
                    if quant_noise_mask is not None:
                        noise = weight_q.data.sub_(x.data).masked_fill_(
                            quant_noise_mask, 0
                        )
                        ### unquantized weights have to follow reparameterization, i.e., tanh and scale
                        weight_q = x + noise
                else:
                    assert NotImplementedError
        else:
            if self.alg == "dorefa":
                weight = torch.tanh(x)  # [-1, 1]
                weight = weight / 2 / torch.max(torch.abs(weight.data)) + 0.5
                # weight = weight / 2 + 0.5
                weight_q = self.uniform_q(weight)
                if quant_noise_mask is not None:
                    noise = weight_q.data.sub_(weight.data).masked_fill_(
                        quant_noise_mask, 0
                    )
                    ### unquantized weights have to follow reparameterization, i.e., tanh and scale
                    weight_q = weight + noise

            elif self.alg == "dorefa_sym":
                weight = torch.tanh(x)  # [-1, 1]
                r = torch.max(torch.abs(weight.data))
                # weight = weight / 2 + 0.5
                weight_q = self.uniform_q(weight / (2 * r) + 0.5) * (2 * r) - r
                if quant_noise_mask is not None:
                    noise = weight_q.data.sub_(weight.data).masked_fill_(
                        quant_noise_mask, 0
                    )
                    ### unquantized weights have to follow reparameterization, i.e., tanh
                    weight_q = weight + noise
            elif self.alg == "dorefa_pos":
                weight = torch.tanh(x)  # [-1, 1]
                r = torch.max(torch.abs(weight.data))
                weight = weight + r
                # weight = weight / 2 + 0.5
                weight_q = self.uniform_q(weight / (2 * r)) * 2 * r
                if quant_noise_mask is not None:
                    noise = weight_q.data.sub_(weight.data).masked_fill_(
                        quant_noise_mask, 0
                    )
                    ### unquantized weights have to follow reparameterization, i.e., tanh
                    weight_q = weight + noise

            elif self.alg == "qnn":
                x_min = torch.min(x.data)
                x_max = torch.max(x.data)
                x_range = x_max - x_min
                weight_q = self.uniform_q((x - x_min) / x_range) * x_range + x_min
                if quant_noise_mask is not None:
                    noise = weight_q.data.sub_(x.data).masked_fill_(quant_noise_mask, 0)
                    ### unquantized weights have to follow reparameterization, i.e., tanh
                    weight_q = x + noise
            else:
                assert NotImplementedError

        return weight_q


class PermutationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, forward_indices):
        ctx.forward_indices = forward_indices
        output = input[..., forward_indices]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        forward_indices = ctx.forward_indices
        grad_input = grad_output.clone()
        grad_input[..., forward_indices] = grad_output
        return grad_input, None


def assert_unitary(x):

    if x.is_complex():
        x_t = x.conj().transpose(-1, -2)
    else:
        x_t = x.transpose(-1, -2)

    x = x_t.matmul(x)
    I = torch.eye(x.size(-1), device=x.device, dtype=x.dtype) + x.mul(0)
    assert torch.allclose(x, I, rtol=1e-3, atol=1e-5), f"{x}"


def get_box(center=(0, 0), size=(5, 2)):
    box = Polygon(
        [
            (center[0] - size[0] / 2, center[1] - size[1] / 2),
            (center[0] + size[0] / 2, center[1] - size[1] / 2),
            (center[0] + size[0] / 2, center[1] + size[1] / 2),
            (center[0] - size[0] / 2, center[1] + size[1] / 2),
        ]
    )
    return box


def get_sinebend(left=(0, 0), right=(1, 1)):
    ## draw a Linestring to form a sine bend with width
    x = np.linspace(left[0], right[0], 50)
    y = (
        abs(right[1] - left[1])
        / 2
        * np.cos(
            (x - left[0]) / (right[0] - left[0]) * np.pi + np.pi * (left[1] < right[1])
        )
        + (right[1] + left[1]) / 2
    )
    # y_up = (
    #     abs(right[1] - left[1]) / 2 * np.cos((x - left[0]) / (right[0] - left[0]) * np.pi + np.pi * (left[1] < right[1]))
    #     + (right[1] + left[1]) / 2
    #     + width / 2
    # )
    # print(y_up)
    # y_low = y_up - width
    # right_edge = [(right[0], right[1] + width / 2), (right[0], right[1] - width / 2)]
    # left_edge = [(left[0], left[1] - width / 2), (left[0], left[1] + width / 2)]
    # up_edge = [(x, y) for x, y in zip(x, y_up)]
    # low_edge = [(x, y) for x, y in zip(x, y_low)][::-1]
    # sinebend = Polygon(up_edge + right_edge + low_edge + left_edge)
    sinebend = (x, y)
    return sinebend


def get_MMI(n_ports=2, center=(0, 0), box_size=(4, 2), wg_gap=10, wg_len=4):
    box = get_box(center=center, size=box_size)
    in_port_left_x = [center[0] - box_size[0] / 2 - wg_len] * n_ports
    in_port_left_y = [
        center[1] - ((n_ports - 1) / 2 - i) * wg_gap for i in range(n_ports)
    ]
    in_port_right_x = [center[0] - box_size[0] / 2] * n_ports
    in_port_right_y = [
        center[1] - ((n_ports - 1) / 2 - i) * box_size[1] / n_ports
        for i in range(n_ports)
    ]
    # print(in_port_left_x, in_port_left_y, in_port_right_x, in_port_right_y)
    in_ports = [
        get_sinebend(
            left=(in_port_left_x[i], in_port_left_y[i]),
            right=(in_port_right_x[i], in_port_right_y[i]),
        )
        for i in range(n_ports)
    ]

    out_port_left_x = [center[0] + box_size[0] / 2] * n_ports
    out_port_left_y = [
        center[1] - ((n_ports - 1) / 2 - i) * box_size[1] / n_ports
        for i in range(n_ports)
    ]
    out_port_right_x = [center[0] + box_size[0] / 2 + wg_len] * n_ports
    out_port_right_y = [
        center[1] - ((n_ports - 1) / 2 - i) * wg_gap for i in range(n_ports)
    ]
    out_ports = [
        get_sinebend(
            left=(out_port_left_x[i], out_port_left_y[i]),
            right=(out_port_right_x[i], out_port_right_y[i]),
        )
        for i in range(n_ports)
    ]

    return [box], in_ports + out_ports


def get_wg(center=(0, 0), wg_len=4):
    x = [center[0] - wg_len / 2, center[0] + wg_len / 2]
    y = [center[1], center[1]]
    return (x, y)


def get_DC_array(
    dc_array=[1, 2, 4, 1], box_size2=(10, 5), center=(0, 0), wg_gap=10, wg_len=20
):
    n_wgs = sum(dc_array)
    ports = []
    boxes = []
    for i, n_ports in enumerate(dc_array):
        start_port_id = int(np.sum(dc_array[:i]))
        if n_ports == 1:
            ports.append(
                get_wg(
                    center=(
                        center[0],
                        center[1]
                        - (n_wgs - 1) / 2 * wg_gap
                        + (start_port_id + (n_ports - 1) / 2) * wg_gap,
                    ),
                    wg_len=wg_len,
                )
            )
        elif n_ports > 1:
            box_size = (
                box_size2[0] * (n_ports / 2) ** 0.7,
                box_size2[1] * (n_ports / 2) ** 0.7,
            )

            box, port = get_MMI(
                n_ports=n_ports,
                center=(
                    center[0],
                    center[1]
                    - (n_wgs - 1) / 2 * wg_gap
                    + (start_port_id + (n_ports - 1) / 2) * wg_gap,
                ),
                box_size=box_size,
                wg_gap=wg_gap,
                wg_len=(wg_len - box_size[0]) / 2,
            )
            boxes += box
            ports += port
    return boxes, ports


def plot_cr_array(cr_array=[0, 3, 2, 1], center=(0, 0), wg_gap=10, wg_len=20):
    ##ce_array is input indices
    n_wgs = len(cr_array)
    left_indices = cr_array
    right_indices = list(range(n_wgs))
    crs = []
    for left, right in zip(left_indices, right_indices):
        crs.append(
            get_sinebend(
                left=(
                    center[0] - wg_len / 2,
                    center[1] - (n_wgs - 1) / 2 * wg_gap + left * wg_gap,
                ),
                right=(
                    center[0] + wg_len / 2,
                    center[1] - (n_wgs - 1) / 2 * wg_gap + right * wg_gap,
                ),
            )
        )
    return crs


def plot_ps_array(n_wgs=8, center=(0, 0), wg_gap=10, wg_len=20, ps_width=4, ps_len=10):
    boxes = []
    ports = []
    for i in range(n_wgs):
        center_tmp = (center[0], center[1] - (n_wgs - 1) / 2 * wg_gap + i * wg_gap)
        box = get_box(center=center_tmp, size=(ps_len, ps_width))
        port = get_wg(center=center_tmp, wg_len=wg_len)
        boxes.append(box)
        ports.append(port)
    return boxes, ports


def plot_mesh_block(
    n_wgs: int = 8,
    cur_x: float = 0,
    dc_array=[1, 2, 4, 1],
    cr_array=[0, 2, 3, 1, 4, 6, 7, 5],
    ps_width: float = 4,
    ps_len: float = 10,
    wg_len: float = 30,
    wg_gap: float = 8,
    mmi2_size: tuple = (10, 5),
) -> None:

    ps_boxes, ps_ports = plot_ps_array(
        n_wgs=n_wgs,
        center=(cur_x, 0),
        wg_gap=wg_gap,
        wg_len=wg_len,
        ps_width=ps_width,
        ps_len=ps_len,
    )
    cur_x += wg_len / 2 + wg_len * n_wgs / 4 / 2
    boxes, ports = get_DC_array(
        dc_array=dc_array,
        box_size2=mmi2_size,
        center=(cur_x, 0),
        wg_gap=wg_gap,
        wg_len=wg_len * 2,
    )
    cur_x += wg_len / 2 + wg_len * n_wgs / 4 / 2
    crs = plot_cr_array(
        cr_array=cr_array, center=(cur_x, 0), wg_gap=wg_gap, wg_len=wg_len
    )
    cur_x += wg_len
    return ps_boxes, boxes, ps_ports + ports + crs, cur_x


def get_MZM(center=(0, 0), box_size=(8, 4), wg_len=30):
    box = Polygon(
        [
            (center[0] - box_size[0] / 2, center[1]),
            (
                center[0] - box_size[0] / 2 + box_size[1] / 2 * 3**0.5,
                center[1] + box_size[1] / 2,
            ),
            (
                center[0] + box_size[0] / 2 - box_size[1] / 2 * 3**0.5,
                center[1] + box_size[1] / 2,
            ),
            (center[0] + box_size[0] / 2, center[1]),
            (
                center[0] + box_size[0] / 2 - box_size[1] / 2 * 3**0.5,
                center[1] - box_size[1] / 2,
            ),
            (
                center[0] - box_size[0] / 2 + box_size[1] / 2 * 3**0.5,
                center[1] - box_size[1] / 2,
            ),
        ]
    )
    # port_len = (wg_len - box_size[0]) / 2
    # port = [
    #     get_wg(center=(center[0] - (box_size[0] + port_len) / 2, center[1]), wg_len=port_len),
    #     get_wg(center=(center[0] + (box_size[0] + port_len) / 2, center[1]), wg_len=port_len),
    # ]
    port = [
        get_wg(center=center, wg_len=wg_len),
    ]
    return [box], port


def get_MZM_array(n_wgs=8, center=(0, 0), wg_gap=8, wg_len=30, mzm_size=(25, 5)):
    boxes = []
    ports = []
    for i in range(n_wgs):
        center_tmp = (center[0], center[1] - (n_wgs - 1) / 2 * wg_gap + i * wg_gap)
        box, port = get_MZM(center=center_tmp, box_size=mzm_size, wg_len=wg_len)
        boxes += box
        ports += port
    return boxes, ports


def plot_mesh(
    filepath: str,
    gene,
    ps_width: float = 4,
    ps_len: float = 20,
    wg_width: float = 0.5,
    wg_len: float = 30,
    wg_gap: float = 8,
    mmi2_size: tuple = (10, 5),
):
    if filepath is not None:
        dir_name = os.path.dirname(filepath)
        ensure_dir(dir_name)

    cur_x = 0
    n_wgs = len(gene[1][1])
    n_blocks = gene[0]
    V = gene[1 : 1 + n_blocks // 2]
    U = gene[1 + n_blocks // 2 : 1 + n_blocks]
    ps_boxes = []
    boxes = []
    ports = []
    for dc_array, cr_array in V:
        dc_array = dc_array.tolist()
        cr_array = cr_array.tolist()

        ps_box, box, port, cur_x = plot_mesh_block(
            n_wgs=n_wgs,
            cur_x=cur_x,
            dc_array=dc_array,
            cr_array=cr_array,
            ps_width=ps_width,
            ps_len=ps_len,
            wg_len=wg_len,
            wg_gap=wg_gap,
            mmi2_size=mmi2_size,
        )
        ps_boxes += ps_box
        boxes += box
        ports += port

    mzm_boxes, mzm_ports = get_MZM_array(
        n_wgs=n_wgs, center=(cur_x, 0), wg_gap=wg_gap, wg_len=wg_len
    )

    boxes += mzm_boxes
    ports += mzm_ports
    cur_x += wg_len

    for dc_array, cr_array in U:
        dc_array = dc_array.tolist()
        cr_array = cr_array.tolist()

        ps_box, box, port, cur_x = plot_mesh_block(
            n_wgs=n_wgs,
            cur_x=cur_x,
            dc_array=dc_array,
            cr_array=cr_array,
            ps_width=ps_width,
            ps_len=ps_len,
            wg_len=wg_len,
            wg_gap=wg_gap,
            mmi2_size=mmi2_size,
        )
        ps_boxes += ps_box
        boxes += box
        ports += port
    boxes = so.unary_union(boxes)
    ps_boxes = so.unary_union(ps_boxes)
    fig, ax = plt.subplots(1, 1, figsize=(4 * n_blocks, 0.4 * n_wgs))
    for geom in boxes.geoms:
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, fc="#1F77B4", ec="none")

    for geom in ps_boxes.geoms:
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, fc="#A31C34", ec="none")

    for bend in ports:
        ax.plot(bend[0], bend[1], linewidth=6 * wg_width, color="#1F77B4")
    # plt.show()
    plt.axis("off")
    plt.tight_layout()
    if filepath is not None:
        plt.savefig(filepath, dpi=300)


if __name__ == "__main__":
    pass
