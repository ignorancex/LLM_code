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
from flash_dropout.cuda.binding_gemm import GEMM
from flash_dropout.functional.blockwise_dropout_matmul_cuda import (
    blockwise_dropout_matmul as blockwise_dropout_matmul_cuda,
)
from flash_dropout.functional.utils import blockwise_dropout_mask
from flash_dropout.triton.dsd_matmul import blockwise_dsd_matmul
from flash_dropout.triton.sdd_matmul import blockwise_sdd_matmul

block_size = 128
CACHE = {}


def f_naive(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    C = F.naive_blockwise_dropout_matmul(A, B, block_size, p)
    yield "forward"

    C.backward(dC)
    yield "backward"


def f_dense(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor):
    C = torch.nn.functional.linear(A, B)
    yield "forward"

    C.backward(dC)
    yield "backward"


def f_baseline(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    C = F.vanilla_dropout_matmul(A, B, p)
    yield "forward"

    C.backward(dC)
    yield "backward"


def f_dense_cuda(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor):
    ext = GEMM()
    C = ext.gemm(A, B)
    yield "forward"

    dA = ext.gemm(dC, B.T)
    dB = ext.gemm(A.T, dC.T).T
    yield "backward"


def f_cuda(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    C = blockwise_dropout_matmul_cuda(A, B, block_size, p)
    yield "forward"

    C.backward(dC)
    yield "backward"


def f_triton_fixed_mask(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    """
    Args:
        A: M K
        B: N K
        dC: M N
    """
    if f"triton_{p}" not in CACHE:
        mask = blockwise_dropout_mask(A, block_size, p)
        mask_T = mask.T
        table = torch.nonzero(~mask, as_tuple=False)
        CACHE[f"triton_{p}"] = (mask, mask_T, table)
    else:
        mask, mask_T, table = CACHE[f"triton_{p}"]
    C = blockwise_dsd_matmul(A, mask, B.T, block_size, 1 / (1 - p))
    yield "forward"

    dA = blockwise_sdd_matmul(dC, B, table, block_size, 1 / (1 - p))
    dB = blockwise_dsd_matmul(A.T, mask_T, dC, block_size, 1 / (1 - p)).T
    yield "backward"


def f_triton(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    """
    Args:
        A: M K
        B: N K
        dC: M N
    """
    mask = blockwise_dropout_mask(A, block_size, p)
    C = blockwise_dsd_matmul(A, mask, B.T, block_size, 1 / (1 - p))
    yield "forward"

    mask_T = mask.T
    table = torch.nonzero(~mask, as_tuple=False)
    dA = blockwise_sdd_matmul(dC, B, table, block_size, 1 / (1 - p))
    dB = blockwise_dsd_matmul(A.T, mask_T, dC, block_size, 1 / (1 - p)).T
    yield "backward"


if __name__ == "__main__":
    import miscellaneous.plot_style

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    args = parser.parse_args()
    M, N, K = args.M, args.N, args.K

    L.seed_everything(0)

    A = make_tensor(M, K)
    B = make_tensor(N, K)
    dC = make_tensor(M, N, requires_grad=False)

    meta = defaultdict(dict)
    fns = {}

    fns["Dense"] = partial(f_dense, A, B, dC)
    meta["Dense"] = {"sparsity": None}

    fns["Dense CUDA"] = partial(f_dense_cuda, A, B, dC)
    meta["Dense CUDA"] = {"sparsity": None}

    # fns["Dropout + Dense"] = partial(f_baseline, A, B, dC, 0.5)
    # meta["Dropout + Dense"] = {"sparsity": None}

    # fns["Block dropout + Dense"] = partial(f_naive, A, B, dC, 0.5)
    # meta["Block dropout + Dense"] = {"sparsity": None}

    for p in np.linspace(0, 0.99, 20):
        fns[f"CUDA {p:.2f}"] = partial(f_cuda, A, B, dC, p)
        meta[f"CUDA {p:.2f}"] = {"sparsity": p, "name": "CUDA"}

        # fns[f"Triton {p:.2f}"] = partial(f_triton, A, B, dC, p)
        # meta[f"Triton {p:.2f}"] = {"sparsity": p, "name": "Triton"}

        # fns[f"Triton (fixed mask) {p:.2f}"] = partial(f_triton_fixed_mask, A, B, dC, p)
        # meta[f"Triton (fixed mask) {p:.2f}"] = {"sparsity": p, "name": "Triton (fixed mask)"}
    df_meta = pd.DataFrame(meta).T

    benchmarker = Benchmarker(fns, warmup_reps=5, duration=10)
    benchmarker.run()
    df = benchmarker.results(time=np.median)
    df = df.join(df_meta, on="method")

    def plot_breakpoint(breakpoint_name: str):
        sub_df = df[df["breakpoint"] == breakpoint_name]
        sparse_df = sub_df[sub_df["sparsity"].notnull()].copy()
        base_flops = 2 * M * N * K if breakpoint_name == "forward" else 4 * M * N * K
        sparse_df["flops"] = (
            base_flops * (1 - sparse_df["sparsity"]) / sparse_df["time"]
        )
        dense_df = sub_df[sub_df["sparsity"].isnull()].copy()
        dense_df["flops"] = base_flops / dense_df["time"]

        def _plot(stat: str):
            fig, ax = plt.subplots(figsize=(5, 4))

            for i, (_, row) in enumerate(dense_df.iterrows()):
                ax.axhline(
                    row[stat], label=row["method"], linestyle="--", color=f"C{i+1}"
                )

            sns.lineplot(
                data=sparse_df,
                x="sparsity",
                y=stat,
                hue="name",
                marker="o",
                markersize=4,
                ax=ax,
            )

            ax.set(xlim=(-0.02, 1.02), ylim=(0, None))

            log_dir = Path("./logs", "full", f"m{M}n{N}k{K}")
            log_dir.mkdir(exist_ok=True, parents=True)
            fig.tight_layout()
            fig.savefig(log_dir / f"{breakpoint_name}_{stat}.png", dpi=300)

        _plot("time")
        _plot("flops")

    for breakpoint_name in df["breakpoint"].unique():
        plot_breakpoint(breakpoint_name)
