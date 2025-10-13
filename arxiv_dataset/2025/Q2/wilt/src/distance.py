import argparse
import copy
import os

import networkx as nx  # type: ignore
import numpy as np  # type: ignore
import ot  # type: ignore
import timeout_decorator  # type: ignore
import torch  # type: ignore
from torch_geometric.data import Data, Dataset  # type: ignore
from torch_geometric.utils import to_networkx  # type: ignore

from path import RESULT_DIR  # type: ignore
from tree import WeisfeilerLemanLabelingTree  # type: ignore
from utils import load_dataset  # type: ignore

NUM_PAIR = 1000


def get_indices(dataset: Dataset) -> np.ndarray:
    # return np.random.RandomState(seed=0).permutation(len(dataset) * len(dataset))[
    #     :NUM_PAIR
    # ]
    n_samples = len(dataset)
    indices_train = np.random.RandomState(seed=0).permutation(n_samples)[
        n_samples // 5 :
    ]
    indices_0 = indices_train[
        np.random.RandomState(seed=1).randint(0, len(indices_train), NUM_PAIR)
    ]
    indices_1 = indices_train[
        np.random.RandomState(seed=2).randint(0, len(indices_train), NUM_PAIR)
    ]
    return indices_0 * n_samples + indices_1


def calc_distance_matrix_WILT_WLOA_WWL(
    dataset: Dataset, dataset_name, metric, **kwargs
) -> torch.Tensor:
    tree = WeisfeilerLemanLabelingTree(
        dataset,
        kwargs["depth"],
        (
            kwargs["normalize"]
            if metric == "WILT"
            else ("dummy" if metric == "WLOA" else "size")
        ),
        False if metric in ["WWL", "WLOA"] else None,
    )
    if metric == "WILT":
        tree.load_parameter(
            os.path.join(
                RESULT_DIR,
                dataset_name,
                kwargs["gnn"],
                f"l={kwargs['n_mp_layers']}_p={kwargs['pooling']}_d={kwargs['emb_dim']}_s={kwargs['gnn_seed']}",
                kwargs["gnn_distance"],
                f"d={kwargs['depth']}_{kwargs['normalize']}_l1={kwargs['l1coeff']}",
                "model_final.pt",
            )
        )
    elif metric == "WLOA":
        tree.parameter = torch.tensor(
            [kwargs["depth"] / 2] + [0.5 for _ in range(len(tree.parameter) - 1)]
        )
    elif metric == "WWL":
        tree.parameter = torch.full((len(tree.parameter),), 1 / (2 * kwargs["depth"]))
    indices = get_indices(dataset)
    embeddings = torch.stack([tree.calc_subtree_weights(g) for g in dataset], dim=0)
    tree.eval()
    distance_matrix = torch.zeros(NUM_PAIR)
    for i in range(NUM_PAIR):
        distance_matrix[i] = tree.calc_distance_between_subtree_weights(
            embeddings[indices[i] // len(dataset)],
            embeddings[indices[i] % len(dataset)],
        )
    return distance_matrix


def calc_distance_matrix_GED(dataset: Dataset, **kwargs) -> torch.Tensor:
    ANS = -1

    @timeout_decorator.timeout(kwargs["timeout"])
    def call_ged(g1, g2):
        g1nx = to_networkx(
            g1,
            node_attrs=["x"],
            edge_attrs=["edge_attr"] if g1.edge_attr is not None else None,
            to_undirected=True,
        )
        g2nx = to_networkx(
            g2,
            node_attrs=["x"],
            edge_attrs=["edge_attr"] if g2.edge_attr is not None else None,
            to_undirected=True,
        )
        nonlocal ANS
        for tmpd in nx.optimize_graph_edit_distance(
            g1nx,
            g2nx,
            lambda x, y: x == y,
            (lambda x, y: x == y) if g1.edge_attr is not None else None,
        ):
            ANS = tmpd

    distance_matrix = torch.zeros(NUM_PAIR)
    indices = get_indices(dataset)
    for i in range(NUM_PAIR):
        g1 = dataset[indices[i] // len(dataset)]
        g2 = dataset[indices[i] % len(dataset)]
        ANS = -1
        try:
            call_ged(g1, g2)
        except:
            pass
        distance_matrix[i] = ANS
    distance_matrix[distance_matrix == -1] = distance_matrix[
        distance_matrix != -1
    ].mean()
    return distance_matrix


def get_neighbors(g: Data) -> dict[int, list[int]]:
    """neighbor indexes for each node

    Args:
        g (Data): input torch_geometric graph

    Returns:
        dict[int, :list[int]]: a dictionary that store the neighbor indexes

    Notes:
        Copied from https://github.com/chingyaoc/TMD
    """
    adj: dict[int, list[int]] = {}
    for i in range(len(g.edge_index[0])):
        node1 = g.edge_index[0][i].item()
        node2 = g.edge_index[1][i].item()
        if node1 in adj.keys():
            adj[node1].append(node2)
        else:
            adj[node1] = [node2]
    return adj


def TMD(g1: Data, g2: Data, dataset_name: str, w: float | list[float], L: int = 4):
    """Tree Mover's Distance between two graphs.

    Args:
        g1 (Data): graph 1
        g2 (Data): graph 2
        dataset_name (str): name of the dataset
        w (float | list[float]): weight constant(s) for each depth
        L (int, optional): Depth of computation trees for calculating TMD. Defaults to 4.

    Returns:
        float: The TMD between g1 and g2

    Notes:
        Copied from https://github.com/chingyaoc/TMD
    """
    if isinstance(w, list):
        assert len(w) == L - 1
    else:
        w = [w] * (L - 1)

    # get attributes
    n1, n2 = len(g1.x), len(g2.x)
    if dataset_name in ["ZINC", "Lipo"]:
        feat1 = torch.zeros(n1, max(torch.max(g1.x).item(), torch.max(g2.x).item()) + 1)  # type: ignore
        feat2 = torch.zeros(n2, max(torch.max(g1.x).item(), torch.max(g2.x).item()) + 1)  # type: ignore
        for i in range(n1):
            feat1[i, g1.x[i].item()] = 1
        for i in range(n2):
            feat2[i, g2.x[i].item()] = 1
    else:
        feat1 = g1.x
        feat2 = g2.x
    adj1 = get_neighbors(g1)
    adj2 = get_neighbors(g2)

    D = np.zeros((n1, n2))

    # level 1 (pair wise distance)
    M = np.zeros((n1 + 1, n2 + 1))
    for i in range(n1):
        for j in range(n2):
            D[i, j] = torch.norm(feat1[i] - feat2[j])
            M[i, j] = D[i, j]
    # distance w.r.t. blank node
    M[:n1, n2] = torch.norm(feat1, dim=1)
    M[n1, :n2] = torch.norm(feat2, dim=1)

    # level l (tree OT)
    for l in range(L - 1):
        M1 = copy.deepcopy(M)
        M = np.zeros((n1 + 1, n2 + 1))

        # calculate pairwise cost between tree i and tree j
        for i in range(n1):
            for j in range(n2):
                try:
                    degree_i = len(adj1[i])
                except:
                    degree_i = 0
                try:
                    degree_j = len(adj2[j])
                except:
                    degree_j = 0

                if degree_i == 0 and degree_j == 0:
                    M[i, j] = D[i, j]
                # if degree of node is zero, calculate TD w.r.t. blank node
                elif degree_i == 0:
                    wass = 0.0
                    for jj in range(degree_j):
                        wass += M1[n1, adj2[j][jj]]
                    M[i, j] = D[i, j] + w[l] * wass
                elif degree_j == 0:
                    wass = 0.0
                    for ii in range(degree_i):
                        wass += M1[adj1[i][ii], n2]
                    M[i, j] = D[i, j] + w[l] * wass
                # otherwise, calculate the tree distance
                else:
                    max_degree = max(degree_i, degree_j)
                    if degree_i < max_degree:
                        cost = np.zeros((degree_i + 1, degree_j))
                        cost[degree_i] = M1[n1, adj2[j]]
                        dist_1, dist_2 = np.ones(degree_i + 1), np.ones(degree_j)
                        dist_1[degree_i] = max_degree - float(degree_i)
                    else:
                        cost = np.zeros((degree_i, degree_j + 1))
                        cost[:, degree_j] = M1[adj1[i], n2]
                        dist_1, dist_2 = np.ones(degree_i), np.ones(degree_j + 1)
                        dist_2[degree_j] = max_degree - float(degree_j)
                    for ii in range(degree_i):
                        for jj in range(degree_j):
                            cost[ii, jj] = M1[adj1[i][ii], adj2[j][jj]]
                    wass = ot.emd2(dist_1, dist_2, cost)

                    # summarize TMD at level l
                    M[i, j] = D[i, j] + w[l] * wass

        # fill in dist w.r.t. blank node
        for i in range(n1):
            try:
                degree_i = len(adj1[i])
            except:
                degree_i = 0

            if degree_i == 0:
                M[i, n2] = torch.norm(feat1[i])
            else:
                wass = 0.0
                for ii in range(degree_i):
                    wass += M1[adj1[i][ii], n2]
                M[i, n2] = torch.norm(feat1[i]) + w[l] * wass

        for j in range(n2):
            try:
                degree_j = len(adj2[j])
            except:
                degree_j = 0
            if degree_j == 0:
                M[n1, j] = torch.norm(feat2[j])
            else:
                wass = 0.0
                for jj in range(degree_j):
                    wass += M1[n1, adj2[j][jj]]
                M[n1, j] = torch.norm(feat2[j]) + w[l] * wass

    # final OT cost
    max_n = max(n1, n2)
    dist_1, dist_2 = np.ones(n1 + 1), np.ones(n2 + 1)
    if n1 < max_n:
        dist_1[n1] = max_n - float(n1)
        dist_2[n2] = 0.0
    else:
        dist_1[n1] = 0.0
        dist_2[n2] = max_n - float(n2)

    wass = ot.emd2(dist_1, dist_2, M)
    return wass


def calc_distance_matrix_TMD(
    dataset: Dataset, dataset_name: str, **kwargs
) -> torch.Tensor:
    distance_matrix = torch.zeros(NUM_PAIR)
    indices = get_indices(dataset)
    ws = [
        None,
        [1 / 1],
        [1 / 2, 2 / 1],
        [1 / 3, 3 / 3, 3 / 1],
        [1 / 4, 4 / 6, 6 / 4, 4 / 1],
        [1 / 5, 5 / 10, 10 / 10, 10 / 5, 5 / 1],
    ]
    for i in range(NUM_PAIR):
        g1 = dataset[indices[i] // len(dataset)]
        g2 = dataset[indices[i] % len(dataset)]
        distance_matrix[i] = TMD(
            g1, g2, dataset_name, ws[kwargs["depth"] - 1], kwargs["depth"]
        )
    return distance_matrix


def calc_distance_matrix(
    dataset_name: str,
    metric: str,
    path: str,
    **kwargs,
) -> None:
    """calculate the distance matrix of the dataset using the specified metric, and save the distance matrix in the specified path.

    Args:
        dataset_name (str): The name of the dataset.
        metric (str): The name of the metric.
        path (str): The path to save the distance matrix.
        **kwargs: additional arguments
    """
    dataset = load_dataset(dataset_name)
    if metric in ["WILT", "WWL", "WLOA"]:
        distance_matrix = calc_distance_matrix_WILT_WLOA_WWL(
            dataset, dataset_name, metric, **kwargs
        )
    elif metric == "GED":
        distance_matrix = calc_distance_matrix_GED(dataset, **kwargs)
    elif metric == "TMD":
        distance_matrix = calc_distance_matrix_TMD(dataset, dataset_name, **kwargs)
    else:
        raise NotImplementedError
    torch.save(distance_matrix, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        choices=[
            "MUTAG",
            "Mutagenicity",
            "IMDB-BINARY",
            "COLLAB",
            "NCI1",
            "ENZYMES",
            "ZINC",
            "Lipo",
        ],
    )
    # Structural Pseudometric
    parser.add_argument("--metric", choices=["WILT", "WWL", "WLOA", "GED", "TMD"])
    parser.add_argument("--depth", type=int, required=False)
    parser.add_argument("--l1coeff", type=float, required=False)
    parser.add_argument(
        "--normalize", type=str, choices=["size", "dummy"], required=False
    )
    parser.add_argument("--timeout", type=int, required=False)
    # GNN
    parser.add_argument(
        "--gnn", type=str, choices=["gcn", "gin", "gat"], required=False
    )
    parser.add_argument("--n_mp_layers", type=int, required=False)
    parser.add_argument("--emb_dim", type=int, required=False)
    parser.add_argument("--pooling", type=str, choices=["sum", "mean"], required=False)
    parser.add_argument("--gnn_seed", type=int, required=False)
    parser.add_argument(
        "--gnn_distance", type=str, choices=["l1", "l2"], required=False
    )
    args = parser.parse_args()
    if args.metric == "WILT":
        path = os.path.join(
            RESULT_DIR,
            args.dataset_name,
            args.gnn,
            f"l={args.n_mp_layers}_p={args.pooling}_d={args.emb_dim}_s={args.gnn_seed}",
            args.gnn_distance,
            f"d={args.depth}_{args.normalize}_l1={args.l1coeff}",
            "WILT.pt",
        )
    else:
        if args.metric in ["WWL", "WLOA"]:
            filepath = f"{args.metric}_d={args.depth}.pt"
        elif args.metric == "GED":
            filepath = f"{args.metric}_t={args.timeout}.pt"
        elif args.metric == "TMD":
            filepath = f"{args.metric}_d={args.depth}.pt"
        path = os.path.join(
            RESULT_DIR,
            args.dataset_name,
            filepath,
        )
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    calc_distance_matrix(**args.__dict__, path=path)
