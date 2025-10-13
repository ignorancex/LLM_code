#!/usr/bin/env python
# train_mnist_ablate.py
# Author: ChatGPT – drop-in ablation sweep for FastKAN on MNIST

import os, sys, json, time, itertools, argparse, csv, pickle, pathlib
from collections import defaultdict
from datetime import datetime

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune

from ptflops import get_model_complexity_info           
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchinfo import summary
from thop import profile

# -------  adjust import to your folder layout -----------------------------
from kan_convs import FastKANConv2DLayer
# ---------------------------------------------------------------------------


# ------------------------------ Model --------------------------------------
class LeNet5_KAN(nn.Module):
    """LeNet-5 skeleton but with FastKANConv2DLayer blocks."""
    def __init__(self, num_classes=10, grid_size=8, width_mult=1.0, use_relu=True):
        super().__init__()
        Act = nn.ReLU if use_relu else nn.Identity
        c1, c2 = int(6 * width_mult), int(16 * width_mult)

        self.layer1 = nn.Sequential(
            FastKANConv2DLayer(
                # conv_class=None, norm_class=None,  # unused in the public impl
                input_dim=1, output_dim=c1,
                kernel_size=5, stride=1, padding=0,
                grid_size=grid_size
            ),
            nn.BatchNorm2d(c1),
            Act(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            FastKANConv2DLayer(
                # conv_class=None, norm_class=None,
                input_dim=c1, output_dim=c2,
                kernel_size=5, stride=1, padding=0,
                grid_size=grid_size,
            ),
            nn.BatchNorm2d(c2),
            Act(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(120),
            Act(),
            nn.Linear(120, 84),
            Act(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.classifier(x)


# ---------------------------- utils ----------------------------------------

def apply_prune(model: nn.Module, amount: float = 0.25):
    """
    Structured L2-norm pruning on the output channels of each Conv inside
    FastKANConv2DLayer.  Skips everything if amount == 0.
    """
    if amount < 1e-6:
        return model

    for mod in model.modules():
        if isinstance(mod, FastKANConv2DLayer):
            # prune every underlying Conv2d in both lists
            for conv in list(mod.base_conv) + list(mod.spline_conv):
                prune.ln_structured(conv, name="weight",
                                    amount=amount, n=2, dim=0)
    return model

def safe_model_metrics(model, input_size=(1, 32, 32), device="cuda"):
    """Return (params, flops) – flops may be None if THOP fails."""
    # 1 – parameter count via torchinfo (always works)
    info = summary(model, input_size=(1, *input_size), verbose=0, depth=0)
    params = info.total_params

    # 2 – FLOPs via thop (robust to unknown layers: it skips them)
    try:
        dummy = torch.randn(1, *input_size).to(device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
    except Exception as e:
        print(f"[warn] THOP failed to compute FLOPs: {e}")
        flops = None
    return params, flops

def ptq_int8(model: nn.Module, example_inp: torch.Tensor):
    """Post-training static INT8 quantisation (simple, fx-graph-mode)."""
    model_cpu = model.cpu().eval()
    qconfig_map = get_default_qconfig_mapping("fbgemm")
    prepared    = prepare_fx(model_cpu, qconfig_map, (example_inp,))
    with torch.inference_mode():
        prepared(example_inp)          # one calibration batch is enough
    return convert_fx(prepared).to(example_inp.device)

def measure_latency(model, device, reps=50):
    model.eval()
    dummy = torch.randn(32, 1, 32, 32, device=device)
    torch.cuda.synchronize()
    start = time.time()
    with torch.inference_mode():
        for _ in range(reps):
            _ = model(dummy)
    torch.cuda.synchronize()
    return (time.time() - start) / reps


# ------------------------- training helpers --------------------------------
def get_loaders(batch=512):
    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.13,), (0.308,))
    ])
    train_ds = datasets.MNIST(root="./mnist/train", train=True, download=False, transform=tfm)
    test_ds = datasets.MNIST(root="./mnist/test", train=False, download=False, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch*2, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, opt, device, loss_fn):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    loss_sum, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        out = model(x)
        loss_sum += loss_fn(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


# ----------------------------- sweep grid ----------------------------------
GRID = {
    "grid_size":  [4, 8, 16],
    "width_mult": [1.0, 1.5],
    "use_relu":   [True, False],
    "prune_amt":  [0.0, 0.25],
    # "quant":      ["fp32", "int8"]
}


# ------------------------------ main ---------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--gpus", type=int, default=1)   # only local single-GPU sweeps here
    parser.add_argument("--outdir", default="mnist_ablate_results")
    args = parser.parse_args()

    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)
    results = []

    # single-GPU only – multi-GPU would require DDP wrapping for every variant
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    train_loader, test_loader = get_loaders()
    loss_fn = nn.CrossEntropyLoss()

    # after train/val loaders are built
    calib_batch, _ = next(iter(train_loader))
    calib_tensor = calib_batch[:32].to(device)   # 32 samples

    for ix, combo in enumerate(itertools.product(*GRID.values())):
        cfg = dict(zip(GRID.keys(), combo))
        tag = "_".join(f"{k}{v}" for k, v in cfg.items() if k != "use_relu")
        print(f"\n=== [{ix+1}/{len(list(itertools.product(*GRID.values())))}] {cfg} ===")
        # ---- model build ---------------------------------------------------
        model = LeNet5_KAN(grid_size=cfg["grid_size"],
                           width_mult=cfg["width_mult"],
                           use_relu=cfg["use_relu"]).to(device)

        # pruning
        model = apply_prune(model, cfg["prune_amt"])

        # quick flop/param count
        params, flops = safe_model_metrics(model, device=device)

        # # then later
        # if cfg["quant"] == "int8":
        #     model = ptq_int8(model, calib_tensor)


        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # ---- train --------------------------------------------------------
        for ep in range(args.epochs):
            tr_loss = train_one_epoch(model, train_loader, opt, device, loss_fn)
            print(f"  ep {ep+1}/{args.epochs}  train_loss={tr_loss:.4f}", end="\r")

        val_loss, val_acc = evaluate(model, test_loader, device, loss_fn)
        latency = measure_latency(model, device, reps=50)

        run_res = {
            **cfg,
            "val_loss": val_loss,
            "val_acc":  val_acc,
            "params": params,
            "flops":    flops,
            "params":   params,
            "latency":  latency
        }
        print(f"→ acc={val_acc:.4f} | loss={val_loss:.3f} | flops={flops/1e6:.1f} M | "
              f"lat {latency*1000:.1f} ms")

        results.append(run_res)
        # checkpoint per run (optional)
        torch.save(model.state_dict(), f"{args.outdir}/model_{tag}.pt")

    # ---------------------- save CSV --------------------------------------
    keys = list(results[0].keys())
    with open(f"{args.outdir}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    # ------------------- quick visualisations -----------------------------
    # 1. acc vs. grid_size (mean over other factors)
    fig = plt.figure(figsize=(6,4))
    grid_vals = sorted(set(r["grid_size"] for r in results))
    mean_acc = [sum(r["val_acc"] for r in results if r["grid_size"]==g)/len([1 for r in results if r["grid_size"]==g])
                for g in grid_vals]
    plt.plot(grid_vals, mean_acc, marker="o")
    plt.ylabel("Validation Accuracy")
    plt.xlabel("grid_size (knots)")
    plt.title("Effect of spline grid_size")
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/acc_vs_grid.pdf")

    # 2. speed-up chart (latency vs. flops)
    # fig = plt.figure(figsize=(5,4))
    # for q in ("fp32","int8"):
    #     xs = [r["flops"]/1e6 for r in results if r["quant"]==q]
    #     ys = [r["latency"]*1000 for r in results if r["quant"]==q]
    #     plt.scatter(xs, ys, label=q, alpha=0.7)
    plt.xlabel("FLOPs (M)")
    plt.ylabel("Latency (ms, batch=32)")
    plt.xscale("log"); plt.yscale("log")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{args.outdir}/lat_vs_flops.pdf")

    # 3. Pareto frontier acc vs. latency
    fig = plt.figure(figsize=(5,4))
    xs = [r["latency"]*1000 for r in results]
    ys = [r["val_acc"] for r in results]
    plt.scatter(xs, ys, c="steelblue")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Val Accuracy")
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/acc_vs_lat.pdf")

    # merge figs
    from PyPDF2 import PdfMerger
    merger = PdfMerger()
    for f in ["acc_vs_grid.pdf","lat_vs_flops.pdf","acc_vs_lat.pdf"]:
        merger.append(f"{args.outdir}/{f}")
    merger.write(f"{args.outdir}/ablation_plots.pdf")
    merger.close()

    print(f"\nAll done!  CSV + plots saved to: {args.outdir}")

if __name__ == "__main__":
    main()

