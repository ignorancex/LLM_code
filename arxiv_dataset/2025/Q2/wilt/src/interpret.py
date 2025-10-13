import argparse
import json
import os

import matplotlib.pyplot as plt  # type: ignore
import torch  # type: ignore

from path import GNN_DIR, RESULT_DIR
from tree import WeisfeilerLemanLabelingTree
from utils import load_dataset


def stat(dataset_name: str, labels: torch.Tensor, predictions: torch.Tensor) -> None:
    if dataset_name in ["synthetic_bin", "Mutagenicity"]:
        n_positive = torch.sum(labels == 1)
        n_negative = torch.sum(labels == 0)
        n_predicted_positive = torch.sum(predictions == 1)
        n_predicted_negative = torch.sum(predictions == 0)
        print(n_positive, n_negative, n_predicted_positive, n_predicted_negative)
    elif dataset_name in ["synthetic_mul", "ENZYMES"]:
        classes = torch.unique(labels)
        n_classes = torch.tensor([torch.sum(labels == c) for c in classes])
        n_predicted = torch.tensor([torch.sum(predictions == c) for c in classes])
        print(n_classes, n_predicted)
    else:
        pass


def color_stat(
    dataset_name: str,
    color: int,
    weight: float,
    rank: int,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    if dataset_name in ["synthetic_bin", "Mutagenicity"]:
        print(
            f"{color}: w={weight} r:{rank}\
            p:{torch.sum(embeddings[labels==1][:,color]>0)}\
            n:{torch.sum(embeddings[labels==0][:,color]>0)}\
            pred_p:{torch.sum(embeddings[predictions==1][:,color]>0)}\
            pred_n:{torch.sum(embeddings[predictions==0][:,color]>0)}"
        )
    elif dataset_name in ["synthetic_mul", "ENZYMES"]:
        classes = torch.unique(labels)
        print(
            f"{color}: w={weight} r:{rank}\
            label: {[torch.sum(embeddings[labels==i][:,color]>0).item() for i in range(len(classes))]}\
            pred: {[torch.sum(embeddings[predictions==i][:,color]>0).item() for i in range(len(classes))]}"
        )
    else:
        print(
            f"{color}: w={weight}\
            r:{rank}\
            corr label:{torch.corrcoef(torch.stack([embeddings[:, color], labels], dim=0))[0, 1]}\
            corr pred: {torch.corrcoef(torch.stack([embeddings[:, color], predictions], dim=0))[0, 1]}"
        )


def interpret(
    dataset_name: str,
    gnn: str,
    n_mp_layers: int,
    emb_dim: int,
    pooling: str,
    gnn_seed: int,
    gnn_distance: str,
    depth: int,
    normalize: str,
    l1coeff: float,
):
    rslt_path = os.path.join(
        RESULT_DIR,
        dataset_name,
        gnn,
        f"l={n_mp_layers}_p={pooling}_d={emb_dim}_s={gnn_seed}",
        gnn_distance,
        f"d={depth}_{normalize}_l1={l1coeff}",
    )
    with open(os.path.join(rslt_path, "info.json")) as f:
        info = json.load(f)

    data = load_dataset(dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth, normalize)
    tree.load_parameter(os.path.join(rslt_path, "model_final.pt"))

    sorted_weights, indices = torch.sort(tree.weight, descending=True)
    embeddings = torch.stack([tree.calc_subtree_weights(g) for g in data], dim=0)

    labels = torch.tensor([g.y.item() for g in data])
    predictions = torch.load(
        os.path.join(
            GNN_DIR,
            dataset_name,
            gnn,
            f"l=3_p={pooling}_d=64_s={gnn_seed}",
            "predictions.pt",
        )
    ).flatten()
    stat(dataset_name, labels, predictions)

    os.makedirs(os.path.join(rslt_path, "interpr"), exist_ok=True)
    cnt = 0
    for idx, c in enumerate(indices):
        if torch.sum(embeddings[:, c] > 0) < len(data) * 0.01:
            continue
        color_stat(
            dataset_name,
            c.item(),
            sorted_weights[idx].item(),
            idx + 1,
            embeddings,
            labels,
            predictions,
        )
        os.makedirs(os.path.join(rslt_path, "interpr", str(c.item())), exist_ok=True)
        tree.unfolding_tree(
            c.item(), os.path.join(rslt_path, "interpr", str(c.item()), "tree.png")
        )
        g_cnt = 0
        for i in range(len(data)):
            # if torch.sum(embeddings[i]) > 30 * 4:
            #     continue
            if embeddings[i, c] > 0:
                os.makedirs(
                    os.path.join(
                        rslt_path, "interpr", str(c.item()), f"{i}_y={data[i].y.item()}"
                    ),
                    exist_ok=True,
                )
                tree.color_unfolding_tree_in_graph(
                    c.item(),
                    data[i],
                    os.path.join(
                        rslt_path, "interpr", str(c.item()), f"{i}_y={data[i].y.item()}"
                    ),
                )
                g_cnt += 1
            if g_cnt >= 3:
                break
        cnt += 1
        if cnt >= 10:
            break

    plt.hist(tree.weight.detach().numpy(), bins=100)
    plt.savefig(os.path.join(rslt_path, "interpr", "weight.png"))
    print(
        f"non zero: {torch.sum(tree.weight > 0)}, zero: {torch.sum(tree.weight == 0)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        choices=[
            "MUTAG",
            "Mutagenicity",
            "NCI1",
            "ENZYMES",
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
    parser.add_argument("--l1coeff", type=float)

    args = parser.parse_args()
    kwargs = args.__dict__
    interpret(**kwargs)
