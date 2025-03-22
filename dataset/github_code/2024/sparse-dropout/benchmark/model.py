import math
import time
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.types
from torch import nn
from torch._tensor import Tensor

from eval.utils import CudaTimer

block_size = (128, 128)


class BenchmarkModel:
    model: nn.Module

    def __init__(self, variant: str, p: float):
        self.fabric = L.Fabric(
            accelerator="auto",
            precision="16-mixed",
        )
        self.fabric.launch()
        self.model = self.fabric.setup_module(self.model)

    def forward(self) -> torch.Tensor:
        raise NotImplementedError()

    def backward(self, loss: torch.Tensor):
        loss.backward()
        # self.fabric.backward(loss)


class MlpModel(BenchmarkModel):
    def __init__(self, variant: str, p: float):
        from eval.mlp.model import BasicNet

        self.x = torch.empty((1024, 1, 32, 32))
        self.y = torch.randint(0, 10, (1024,), dtype=torch.long)
        self.model = BasicNet(
            self.x,
            num_layers=2,
            hidden_dim=1024,
            output_dim=10,
            dropout=dict(
                variant=variant,
                p=p,
                block_size=(128, 128),
            ),
        )
        super().__init__(variant, p)
        self.x = self.fabric.to_device(self.x)
        self.y = self.fabric.to_device(self.y)

    def forward(self) -> Tensor:
        return torch.nn.functional.cross_entropy(self.model(self.x), self.y)


class ViTModel(BenchmarkModel):
    def __init__(self, variant: str, p: float):
        from eval.vit.model import ViT

        self.x = torch.empty((64, 1, 32, 32))
        self.y = torch.randint(0, 10, (64,), dtype=torch.long)
        self.model = ViT(
            (self.x, self.y),
            num_classes=10,
            patch_size=(2, 2),
            block_size=(8, 8),
            n_embed=1024,
            n_head=8,
            n_layers=2,
            dropout=dict(
                variant=variant,
                p=p,
                block_size=(128, 128),
            ),
        )
        super().__init__(variant, p)

        self.x = self.fabric.to_device(self.x)
        self.y = self.fabric.to_device(self.y)

    def forward(self) -> Tensor:
        return torch.nn.functional.cross_entropy(self.model(self.x), self.y)


class LlmModel(BenchmarkModel):
    def __init__(self, variant: str, p: float):
        from eval.llm.model import GPT

        self.x = torch.randint(0, 65, (64, 128), dtype=torch.long)
        self.model = GPT(
            vocab_size=65,
            context_length=128,
            n_layer=4,
            n_head=8,
            n_embed=1024,
            dropout=dict(
                variant=variant,
                p=p,
                block_size=(128, 128),
            ),
        )
        super().__init__(variant, p)

        self.x = self.fabric.to_device(self.x)

    def forward(self) -> Tensor:
        _, loss = self.model(self.x)
        return loss


def do_bench_detailed(model: BenchmarkModel, warmup=0.5, rep=0.2):
    def f():
        loss = model.forward()
        model.backward(loss)

    cache = torch.empty(int(4e6 // 8), dtype=torch.int64, device="cuda")

    for _ in range(10):
        f()

    start_time = time.time()
    while time.time() - start_time < warmup:
        f()
        torch.cuda.synchronize()

    timings = []
    start_time = time.time()
    while time.time() - start_time < rep:
        cache.zero_()
        with CudaTimer() as timer:
            f()
        timings.append(timer.elapsed())

    return timings


if __name__ == "__main__":
    import miscellaneous.plot_style

    L.seed_everything(0)

    models = {
        "MLP": MlpModel,
        # "ViT": ViTModel,
        # "LLM": LlmModel,
    }

    variant_name = {
        "vanilla": "Dropout + Dense",
        "none": "Dense",
    }

    dense_data = []
    for name, model_cls in models.items():
        for variant in ["vanilla", "none"]:
            model = model_cls(variant, p=0.5)
            timings = do_bench_detailed(model, warmup=0.2, rep=0.2)
            dense_data.append(
                {
                    "name": name,
                    "variant": variant,
                    "avg": np.mean(timings),
                    "delta": 1.96 * np.std(timings) / np.sqrt(len(timings)),
                }
            )
    dense_df = pd.DataFrame(dense_data)
    print(dense_df)

    sparse_data = []
    for name, model_cls in models.items():
        for p in np.arange(0, 1, 0.05):
            model = model_cls("blockwise[cuda]", p=p)
            timings = do_bench_detailed(model, warmup=0.2, rep=0.2)
            for t in timings:
                sparse_data.append(
                    {
                        "name": name,
                        "p": p,
                        "time": t,
                    }
                )
    sparse_df = pd.DataFrame(sparse_data)
    print(sparse_df.groupby(["name", "p"])["time"].agg(["mean", "sem"]))

    for name in models:
        fig, ax = plt.subplots(figsize=(5, 4))

        sub_dense_df = dense_df[dense_df["name"] == name]
        sub_sparse_df = sparse_df[sparse_df["name"] == name]

        for i, variant in enumerate(sub_dense_df["variant"].unique()):
            dense_data = sub_dense_df[sub_dense_df["variant"] == variant]
            avg = dense_data["avg"].item()
            delta = dense_data["delta"].item()
            ax.axhline(
                avg, label=variant_name[variant], linestyle="--", color=f"C{i+1}"
            )
            ax.fill_between(
                [-0.02, 1.02],
                [avg - delta, avg - delta],
                [avg + delta, avg + delta],
                color=f"C{i+1}",
                alpha=0.2,
            )
        sns.lineplot(
            data=sub_sparse_df,
            x="p",
            y="time",
            marker="o",
            markersize=4,
            label="SparseDrop",
            ax=ax,
        )

        ax.set(
            ylabel="Time (ms)",
            xlabel="Sparsity",
            xlim=(-0.02, 1.02),
        )
        fig.tight_layout()
        fig.savefig(f"./logs/benchmark_{name}.png", dpi=300)
