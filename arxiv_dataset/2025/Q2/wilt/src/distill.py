import argparse
import json
import os
import random
import time

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from torch.optim import Adam  # type: ignore
from torch.utils.data import BatchSampler  # type: ignore
from torch_geometric.data import Dataset  # type: ignore

from path import GNN_DIR, RESULT_DIR  # type: ignore
from tree import WeisfeilerLemanLabelingTree
from utils import calc_rmse_wo_outliers, load_dataset


class PairSampler(BatchSampler):
    """Sampler for pairwise data

    Attributes:
        dataset (Dataset): dataset to sample from
        batch_size (int): batch size
        distances (torch.Tensor): distance matrix for supervision
        train (bool): whether for training or not
        all_pairs (torch.Tensor): all pairs of indices
    """

    def __init__(
        self, dataset: Dataset, batch_size: int, distances: torch.Tensor, train: bool
    ) -> None:
        """initialize the sampler

        Args:
            dataset (Dataset): dataset
            batch_size (int): batch size
            distances (torch.Tensor): distance matrix for supervision
            train (bool): whether for training or not
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.distances = distances
        self.train = train
        n_samples = len(dataset)
        all_pairs = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                all_pairs.append((i, j))
        self.all_pairs = torch.tensor(all_pairs)  # (nC2, 2)

    def __iter__(self):
        if self.train:
            indices = torch.randperm(len(self.all_pairs))
        else:
            indices = torch.arange(len(self.all_pairs))
        for i in range(0, len(self.all_pairs), self.batch_size):
            batch_pairs = self.all_pairs[indices[i : i + self.batch_size]]
            yield batch_pairs[:, 0], batch_pairs[:, 1], self.distances[
                batch_pairs[:, 0], batch_pairs[:, 1]
            ]

    def __len__(self):
        return (len(self.all_pairs) + self.batch_size - 1) // self.batch_size


def distance_scatter_plot(
    tree: WeisfeilerLemanLabelingTree,
    sampler: PairSampler,
    subtree_weights: torch.Tensor,
    path: str,
) -> tuple:
    """scatter plot of (approximated distance, ground truth distance)

    Args:
        tree (WeisfeilerLemanLabelingTree): WILT
        sampler (PairSampler): sampler for pairwise data
        subtree_weights (torch.Tensor): subtree weights
        path (str): path to save the plot

    Returns:
        tuple: RMSE(d_MPNN, d_WILT), coeff
    """
    tree.eval()
    ys_list = []
    preds_list = []
    max_pair = 10000
    for left_indices, right_indices, y in sampler:
        left_weights = subtree_weights[left_indices]
        right_weights = subtree_weights[right_indices]
        prediction = tree.calc_distance_between_subtree_weights(
            left_weights, right_weights
        )
        ys_list.append(y)
        preds_list.append(prediction)
        if len(ys_list) * len(ys_list[0]) > max_pair:
            break
    ys = torch.cat(ys_list)
    preds = torch.cat(preds_list)
    _, _, coeff, rmse = calc_rmse_wo_outliers(preds.numpy().copy(), ys.numpy().copy())
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["text.usetex"] = True
    plt.scatter(preds, ys)
    fitx = np.linspace(0, torch.max(preds).item(), 100)
    fity = coeff * fitx * (torch.max(ys).item() / torch.max(preds).item())
    plt.plot(fitx, fity, color="red")
    plt.xlabel(r"$d_\mathrm{WILT}$")
    plt.ylabel(r"$d_\mathrm{MPNN}$")
    plt.title(f"RMSE: {rmse:.3f}, coeff: {coeff:.3f}")
    plt.savefig(path)
    plt.close()
    return rmse, coeff


def train_gd(
    data: Dataset,
    tree: WeisfeilerLemanLabelingTree,
    seed: int,
    path: str,
    l1coeff: float,
    batch_size: int,
    n_epochs: int,
    lr: float,
    save_interval: int,
    distances: torch.Tensor,
) -> None:
    """train the model with gradient descent

    Args:
        data (Dataset): training dataset
        tree (WeisfeilerLemanLabelingTree): WILT
        seed (int): random seed
        path (str): path to the directory to save the results
        l1coeff (float): coefficient for L1 regularization
        batch_size (int): batch size
        n_epochs (int): number of epochs
        lr (float): learning rate
        save_interval (int): How often to save the model
        distances (torch.Tensor): distance matrix for training
    """

    # fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # prepare sampler, loss function, and optimizer
    sampler = PairSampler(data, batch_size, distances, train=True)
    loss_fn = torch.nn.MSELoss()
    optimizer = Adam([tree.parameter], lr=lr)

    os.makedirs(path, exist_ok=True)

    # train the model
    loss_hist = []
    epoch_time: float = 0
    subtree_weights = torch.stack([tree.calc_subtree_weights(g) for g in data], dim=0)
    for epoch in range(n_epochs):
        # training
        tree.train()
        start = time.time()
        loss_sum = 0
        for i, (left_indices, right_indices, y) in enumerate(sampler):
            left_weights = subtree_weights[left_indices]
            right_weights = subtree_weights[right_indices]
            prediction = tree.calc_distance_between_subtree_weights(
                left_weights, right_weights
            )
            loss = (
                loss_fn(prediction, y) + torch.sum(torch.abs(tree.parameter)) * l1coeff
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tree.parameter.data = torch.clamp(tree.parameter, min=0)
            loss_sum += loss.item() * len(y)
            if (i + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}, Train batch {i + 1}/{len(sampler)}")
        loss_hist.append(loss_sum / len(sampler.all_pairs))
        end = time.time()
        epoch_time += end - start

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Train loss: {loss_hist[-1]}")
        if (epoch + 1) % save_interval == 0:
            torch.save(tree.parameter, os.path.join(path, f"model_{epoch + 1}.pt"))
    epoch_time /= n_epochs

    # save the loss plot
    plt.plot(loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(path, "loss.png"))
    plt.close()

    # scatter plots
    rmse, coeff = distance_scatter_plot(
        tree,
        sampler,
        subtree_weights,
        os.path.join(path, "scatter.png"),
    )

    # save the training information
    info = {
        "epoch_time": epoch_time,
        "loss_history": loss_hist,
        "rmse": str(rmse),
        "coeff": str(coeff),
    }
    with open(os.path.join(path, "rslt.json"), "w") as f:
        json.dump(info, f)

    # save the model
    torch.save(tree.parameter, os.path.join(path, "model_final.pt"))


def train_wrapper(
    dataset_name: str,
    depth: int,
    normalize: str,
    seed: int,
    gnn: str,
    n_mp_layers: int,
    emb_dim: int,
    pooling: str,
    gnn_seed: int,
    gnn_distance: str,
    path: str,
    **kwargs,
):
    """wrapper function for training

    Args:
        dataset_name (str): dataset name
        depth (int): number of layers in the WILT
        normalize (str): normalization method
        seed (int): random seed
        gnn (str): GNN model
        n_mp_layers (int): number of message passing layers
        emb_dim (int): embedding dimension
        pooling (str): pooling method
        gnn_seed (int): random seed for GNN
        gnn_distance (str): distance metric for GNN embeddings
        path (str): path to the directory to save the results
        **kwargs: additional arguments
    """

    data = load_dataset(dataset_name)
    tree_start = time.time()
    tree = WeisfeilerLemanLabelingTree(data, depth, normalize)
    tree_end = time.time()

    if os.path.exists(os.path.join(path, "rslt.json")):
        print(f"{path} already exists")
        raise ValueError("Already exists")
    distances = torch.load(
        os.path.join(
            GNN_DIR,
            f"{dataset_name}",
            f"{gnn}",
            f"l={n_mp_layers}_p={pooling}_d={emb_dim}_s={gnn_seed}",
            f"dist_{gnn_distance}_last.pt",
        )
    ).to(torch.float32)
    train_gd(
        data,
        tree,
        seed,
        path,
        kwargs["l1coeff"],
        kwargs["batch_size"],
        kwargs["n_epochs"],
        kwargs["lr"],
        kwargs["save_interval"],
        distances,
    )
    tree.reset_parameter()

    # save the training information
    info = {
        "dataset_name": dataset_name,
        "depth": depth,
        "normalize": normalize,
        "seed": seed,
        "gnn": gnn,
        "n_mp_layers": n_mp_layers,
        "emb_dim": emb_dim,
        "pooling": pooling,
        "gnn_seed": gnn_seed,
        "gnn_distance": gnn_distance,
        "tree_time": tree_end - tree_start,
        "l1coeff": kwargs["l1coeff"],
        "batch_size": kwargs["batch_size"],
        "n_epochs": kwargs["n_epochs"],
        "lr": kwargs["lr"],
        "save_interval": kwargs["save_interval"],
    }
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "info.json"), "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        choices=[
            "MUTAG",
            "Mutagenicity",
            "NCI1",
            "ENZYMES",
            "IMDB-BINARY",
            "COLLAB",
            "synthetic_bin",
            "synthetic_mul",
            "synthetic_reg",
            "ZINC",
            "Lipo",
            "ESOL",
        ],
    )
    # GNN
    parser.add_argument("--gnn", choices=["gcn", "gin", "gat"])
    parser.add_argument("--n_mp_layers", type=int)
    parser.add_argument("--emb_dim", type=int)
    parser.add_argument("--pooling", type=str, choices=["sum", "mean"])
    parser.add_argument("--gnn_seed", type=int)
    parser.add_argument("--gnn_distance", type=str, choices=["l1", "l2"])
    # WILT
    parser.add_argument("--depth", type=int)
    parser.add_argument("--normalize", type=str, choices=["size", "dummy"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l1coeff", type=float)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--save_interval", type=int, default=10)

    args = parser.parse_args()
    kwargs = args.__dict__

    kwargs["path"] = os.path.join(
        RESULT_DIR,
        args.dataset_name,
        args.gnn,
        f"l={args.n_mp_layers}_p={args.pooling}_d={args.emb_dim}_s={args.gnn_seed}",
        args.gnn_distance,
        f"d={args.depth}_{args.normalize}_l1={args.l1coeff}",
    )

    if os.path.exists(os.path.join(kwargs["path"], "info.json")):
        print(f"{kwargs['path']} already exists")
        exit()
    os.makedirs(kwargs["path"], exist_ok=True)
    train_wrapper(**kwargs)
