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

from benchmark.utils import Benchmarker, make_tensor
from flash_dropout.cuda.binding_gemm import GEMM
from flash_dropout.functional.utils import blockwise_dropout_mask

block_size = 128
CACHE = {}


def f_dense(A: torch.Tensor, B: torch.Tensor):
    C = torch.nn.functional.linear(A, B)
    yield "forward"


def f_dense_cuda(A: torch.Tensor, B: torch.Tensor):
    ext = GEMM()
    C = ext.gemm(A, B)
    yield "forward"


def f_cuda_cached(A: torch.Tensor, B: torch.Tensor, p: float):
    ext = GEMM()
    if f"cuda_{p}" not in CACHE:
        CACHE[f"cuda_{p}"] = blockwise_dropout_mask(A, block_size, p)
    mask = CACHE[f"cuda_{p}"]
    C = ext.gemm_dsd(A, B, mask, block_size, 1 / (1 - p))
    yield "forward"


def f_cuda(A: torch.Tensor, B: torch.Tensor, p: float):
    ext = GEMM()
    mask = blockwise_dropout_mask(A, block_size, p)
    C = ext.gemm_dsd(A, B, mask, block_size, 1 / (1 - p))
    yield "forward"


if __name__ == "__main__":
    import miscellaneous.plot_style

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("A_layout", type=str, default="row", choices=["row", "col"])
    parser.add_argument("B_layout", type=str, default="col", choices=["row", "col"])
    args = parser.parse_args()
    M, N, K = args.M, args.N, args.K

    L.seed_everything(0)

    A = make_tensor(M, K, args.A_layout == "row")
    B = make_tensor(N, K, args.B_layout == "row")

    meta = defaultdict(dict)
    fns = {}

    fns["Dense"] = partial(f_dense, A, B)
    fns["Dense CUDA"] = partial(f_dense_cuda, A, B)

    meta["Dense"] = {"sparsity": None}
    meta["Dense CUDA"] = {"sparsity": None}

    for p in np.linspace(0, 0.99, 20):
        fns[f"CUDA {p:.2f}"] = partial(f_cuda, A, B, p)
        meta[f"CUDA {p:.2f}"] = {"sparsity": p, "name": "CUDA"}

        # fns[f"CUDA (cached) {p:.2f}"] = partial(f_cuda_cached, A, B, p)
        # meta[f"CUDA (cached) {p:.2f}"] = {"sparsity": p, "name": "CUDA (cached)"}
    df_meta = pd.DataFrame(meta).T

    benchmarker = Benchmarker(fns, warmup_reps=5, duration=5)
    benchmarker.run()
    df = benchmarker.results(time=np.median)
    df = df.join(df_meta, on="method")

    def plot_breakpoint(breakpoint_name: str):
        sub_df = df[df["breakpoint"] == breakpoint_name]
        sparse_df = sub_df[sub_df["sparsity"].notnull()].copy()
        sparse_df["flops"] = (
            2 * M * N * K * (1 - sparse_df["sparsity"]) / sparse_df["time"]
        )
        dense_df = sub_df[sub_df["sparsity"].isnull()].copy()
        dense_df["flops"] = 2 * M * N * K / dense_df["time"]

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

            log_dir = Path(
                "./logs", "dsd", f"m{M}n{N}k{K}", f"{args.A_layout}_{args.B_layout}"
            )
            log_dir.mkdir(exist_ok=True, parents=True)
            fig.tight_layout()
            fig.savefig(log_dir / f"{breakpoint_name}_{stat}.png", dpi=300)

        _plot("time")
        _plot("flops")

    for breakpoint_name in df["breakpoint"].unique():
        plot_breakpoint(breakpoint_name)
