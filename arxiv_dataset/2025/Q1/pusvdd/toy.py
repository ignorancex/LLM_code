import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

import dataloaders
from models import detectors, losses, trainers, utils


class Config(argparse.Namespace):
    algorithm: str
    alpha: float
    n_epoch: int
    learning_rate: float
    batch_size: int
    seed: int


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="AE")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    config = parser.parse_args(namespace=Config())

    if config.algorithm not in ["PU", "PUAE", "PUSVDD", "LOE", "SOEL"]:
        config.alpha = 0.0

    print("Setting:", config)

    # Key
    key = "_".join([str(v) for v in vars(config).values()])

    # Seed
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset
    n_normal = 900
    n_unlabeled_anomaly = 80
    n_labeled_anomaly = 20

    train_loader = dataloaders.make_toy_data(
        batch_size=config.batch_size,
        n_normal=n_normal,
        n_unlabeled_anomaly=n_unlabeled_anomaly,
        n_labeled_anomaly=n_labeled_anomaly,
        is_train=True,
    )
    valid_loader = dataloaders.make_toy_data(
        batch_size=config.batch_size,
        n_normal=n_normal,
        n_unlabeled_anomaly=n_unlabeled_anomaly,
        n_labeled_anomaly=n_labeled_anomaly,
        is_train=True,
    )
    test_loader = dataloaders.make_toy_data(
        batch_size=config.batch_size,
        n_normal=n_normal,
        n_unlabeled_anomaly=n_unlabeled_anomaly,
        n_labeled_anomaly=n_labeled_anomaly,
        is_train=False,
    )

    # Model
    n_in = 2
    n_latent = 2
    n_h = 500

    model = detectors.load_dense_model(algorithm=config.algorithm, n_in=n_in, n_latent=n_latent, n_h=n_h, epsilon=0.1)
    model = model.to(device)
    print("Model:", type(model))

    # Criterion
    criterion = losses.load(algorithm=config.algorithm, alpha=config.alpha)

    # Train
    for path in ["checkpoints", "results", "images"]:
        os.makedirs(path, exist_ok=True)

    if isinstance(model, detectors.PreTrainableSVDD):
        # Pre-Training DeepSVDD model as Autoencoder
        print("Pre-Training DeepSVDD:")
        pre_trainer = trainers.Trainer(device)
        pre_trainer.fit(
            model=model,
            criterion=losses.AELoss(),
            train_loader=train_loader,
            valid_loader=valid_loader,
            checkpoint=f"checkpoints/{key}.pt",
            n_epoch=config.n_epoch,
            learning_rate=config.learning_rate,
            weight_decay=1e-3,
        )

    if isinstance(model, detectors.DeepSVDD):
        # Set center for DeepSVDD model
        print("Set center for DeepSVDD:")
        utils.set_center(model=model, train_loader=train_loader, device=device, eps=0.1)

    print("Training:")
    trainer = trainers.Trainer(device=device)
    trainer.fit(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        checkpoint=f"checkpoints/{key}.pt",
        n_epoch=config.n_epoch,
        learning_rate=config.learning_rate,
        weight_decay=1e-3,
    )

    # Test
    test_score = utils.compute_auc(model=model, test_loader=test_loader, device=device)
    print("AUC:", test_score)

    result = utils.Result(
        train_loss=trainer.train_losses,
        valid_loss=trainer.valid_losses,
        test_score=test_score,
        seen_anomaly_score=0,
        unseen_anomaly_score=0,
    )

    # Save and Plot Results
    utils.save_result(result=result, file_name=f"results/{key}.json")
    utils.plot_result(result=result, file_name=f"images/{key}.pdf")

    # Plot
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(
        {
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "axes.labelsize": 20,
            "legend.fontsize": 20,
            "ps.useafm": True,
            "pdf.use14corefonts": True,
            "text.usetex": True,
            "font.family": "Times New Roman",
        }
    )

    colors = [
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

    xx = np.linspace(-7.5, 7.5, 100)
    yy = np.linspace(-7.5, 7.5, 100)
    XX, YY = np.meshgrid(xx, yy)
    PP = torch.tensor(np.dstack((XX, YY)).astype(np.float32).reshape(-1, 2))

    with torch.no_grad():
        ZZ = model.estimate(PP.to(device)).cpu().numpy().reshape(100, 100)

    plt.clf()
    fig, ax = plt.subplots()
    mappable = ax.contourf(XX, YY, ZZ, levels=20, cmap="viridis")
    fig.colorbar(mappable, format="%.2f")

    X, y = train_loader.dataset.tensors  # type: ignore
    X = X.numpy()
    y = y.numpy()

    X_unlabeled_normal = X[:n_normal]
    X_unlabeled_anomaly = X[n_normal : n_normal + n_unlabeled_anomaly]
    X_labeled_anomaly = X[-n_labeled_anomaly:]

    plt.scatter(
        X_unlabeled_normal[:, 0],
        X_unlabeled_normal[:, 1],
        color=colors[1],
        label="Unlabeled Normal",
    )
    plt.scatter(
        X_unlabeled_anomaly[:, 0],
        X_unlabeled_anomaly[:, 1],
        color=colors[3],
        label="Unlabeled Anomaly",
    )
    plt.scatter(
        X_labeled_anomaly[:, 0],
        X_labeled_anomaly[:, 1],
        color=colors[0],
        label="Labeled Anomaly",
    )

    # Plot unseen anomaly
    plt.scatter(
        [-5, 5],
        [-5, 5],
        marker="*",
        s=600,
        color=colors[4],
        edgecolors="white",
        label="Unseen Anomaly",
    )

    if config.algorithm == "PU":
        plt.legend(
            loc="upper left",
            handletextpad=0.15,
            borderpad=0.15,
            borderaxespad=0.15,
            framealpha=0.5,
        )

    plt.savefig(f"images/{key}_heatmap.pdf")
