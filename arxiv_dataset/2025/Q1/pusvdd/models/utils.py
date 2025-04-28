import json
from dataclasses import asdict, dataclass
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from models import detectors


@dataclass
class Result:
    train_loss: list[float]
    valid_loss: list[float]
    test_score: float
    seen_anomaly_score: float
    unseen_anomaly_score: float


def compute_auc(model: detectors.Detector, test_loader: DataLoader, device: torch.device) -> float:
    y_score = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            s = model.estimate(x)
            y_score.extend(s.cpu().tolist())
            y_true.extend(y.cpu().tolist())

    return roc_auc_score(y_true=y_true, y_score=y_score)  # type: ignore


def save_result(result: Result, file_name: str) -> None:
    with open(file_name, "w") as f:
        f.write(json.dumps(asdict(result)))


def load_result(file_name: str) -> Result:
    with open(file_name, "r") as f:
        return Result(**json.loads(f.read()))


def plot_result(result: Result, file_name: str) -> None:
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(
        {
            "figure.titlesize": 18,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "axes.labelsize": 18,
            "legend.fontsize": 15,
        }
    )

    colors: Final = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    plt.clf()
    plt.plot(
        np.arange(len(result.train_loss)),
        result.train_loss,
        label="Train",
        color=colors[0],
    )
    plt.plot(
        np.arange(len(result.valid_loss)),
        result.valid_loss,
        label="Valid",
        color=colors[1],
    )
    plt.legend()

    plt.suptitle(f"Test Score: {result.test_score:.3f}")
    plt.tight_layout()

    plt.savefig(file_name)


def set_center(
    model: detectors.DeepSVDD,
    train_loader: DataLoader,
    device: torch.device,
    eps: float = 0.1,
) -> None:
    N_train = len(train_loader.dataset)  # type: ignore
    n_latent = 128 if isinstance(model, detectors.ConvolutionalSVDD) else model.n_latent
    c = torch.zeros(n_latent, device=device)

    # compute center
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            x = batch[0].to(device)
            output = model.encode(x)
            c += torch.sum(output, dim=0) / N_train

    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    # set center
    model.set_center(c)
