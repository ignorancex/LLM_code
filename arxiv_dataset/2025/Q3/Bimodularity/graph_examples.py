from typing import Optional
import numpy as np


def random_directed_edges_id(
    ids: list,
    edge_prob: float = 0.5,
    directed: bool = True,
    out_prob: float = 0.5,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random edges from a list of ids

    Parameters
    ----------
    ids : list
        list of edge ids
    edge_prob : float, optional
        probability to select an edge, by default 0.5
    directed : bool, optional
        condition for directed connections, by default True
    out_prob : float, optional
        probability for an out-going edge (only in the directed case), by default 0.5
    seed : Optional[int], optional
        seed for reproducibility, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ids of the selected edges (tuple of outgoing and incoming if directed)
    """

    n_edges = np.shape(ids)[-1]

    np.random.seed(seed)
    edge_ids = np.random.choice(
        np.arange(n_edges), int(n_edges * edge_prob), replace=False
    )

    if directed:
        out_ids = np.random.choice(
            edge_ids, int(len(edge_ids) * out_prob), replace=False
        )
        in_ids = np.array([i for i in edge_ids if i not in out_ids])

        return ids[:, list(out_ids)], ids[:, list(in_ids)]
    return ids[:, edge_ids]


def random_graph(
    n_nodes: int, edge_prob=0.5, directed=False, seed: Optional[int] = None
) -> np.ndarray:
    """Generate a random graph with a given number of nodes

    Parameters
    ----------
    n_nodes : int
        number of nodes
    edge_prob : float, optional
        probability for an edge to connect two nodes, by default 0.5
    directed : bool, optional
        condition for directed connections, by default False
    seed : Optional[int], optional
        seed for reproducibility, by default None

    Returns
    -------
    np.ndarray
        adjacency matrix of the generated graph
    """

    a_mat = np.zeros((n_nodes, n_nodes), dtype=int)

    all_ids = np.array(np.triu_indices_from(a_mat, k=1))

    edge_ids = random_directed_edges_id(
        all_ids, edge_prob=edge_prob, directed=directed, out_prob=0.5, seed=seed
    )

    if directed:
        edge_ids = np.hstack([edge_ids[0], np.flip(edge_ids[1], axis=0)])
    else:
        edge_ids = np.hstack([edge_ids, np.flip(edge_ids, axis=0)])

    a_mat[*edge_ids] = 1

    return a_mat


def random_connector(
    n_nodes: int,
    edge_prob: float = 0.5,
    directed: bool = True,
    out_prob: float = 0.5,
    seed: Optional[float] = None,
) -> np.ndarray:
    """Generate a random "connector" graph with a given number of nodes

    Parameters
    ----------
    n_nodes : int
        number of nodes
    edge_prob : float, optional
        probability for an edge to connect two nodes, by default 0.5
    directed : bool, optional
        condition for directed connections, by default True
    out_prob : float, optional
        probability for an out-going edge (only in the directed case), by default 0.5
    seed : Optional[float], optional
        seed for reproducibility, by default None

    Returns
    -------
    np.ndarray
        adjacency matrix (two matrices in the directed case) of the generated
        "connector" graph
    """

    all_ids = np.array([(i, j) for i in range(n_nodes) for j in range(n_nodes)]).T

    edge_ids = random_directed_edges_id(
        all_ids, edge_prob=edge_prob, directed=directed, out_prob=out_prob, seed=seed
    )

    if directed:
        a_mat = np.zeros((2, n_nodes, n_nodes), dtype=int)
        a_mat[0][*edge_ids[0]] = 1
        # Transpose the in edges
        a_mat[1][edge_ids[1][1], edge_ids[1][0]] = 1
    else:
        a_mat = np.zeros((n_nodes, n_nodes), dtype=int)
        a_mat[edge_ids] = 1

    return a_mat


def toy_random(
    n_nodes: int,
    out_prob=0.5,
    edge_prob=0.5,
    directed=False,
    con_prob: Optional[float] = None,
) -> np.ndarray:
    """Generate a random graph with a given number of nodes. The graph will have two
    densily connected communities (density 'edge_prob') with a random number of
    connecting edges (desntiy 'con_prob').

    Parameters
    ----------
    n_nodes : int
        number of nodes
    out_prob : float, optional
        probability for an out-going edge (only in the directed case), by default 0.5
    edge_prob : float, optional
        probability for an edge to connect two nodes, by default 0.5
    directed : bool, optional
        condition for directed connections, by default False
    con_prob : Optional[float], optional
        density of connecting edges, by default None

    Returns
    -------
    np.ndarray
        graph adjacency matrix
    """
    half_nodes = int(n_nodes / 2)

    com_1 = random_graph(n_nodes=half_nodes, edge_prob=edge_prob, directed=directed)
    com_2 = random_graph(n_nodes=half_nodes, edge_prob=edge_prob, directed=directed)

    if con_prob is None:
        con_prob = edge_prob / 2
    n_connect = half_nodes**2
    edge_idx = np.random.choice(n_connect, int(n_connect * con_prob), replace=False)

    if directed:
        chosen_out_edges = np.random.choice(
            edge_idx,
            int(len(edge_idx) * out_prob),
            replace=False,
        )
        edge_out_idx = np.array([i in chosen_out_edges for i in edge_idx])
        edge_out = edge_idx[edge_out_idx]
        edge_in = edge_idx[~edge_out_idx]
    else:
        edge_out = edge_idx.copy()
        edge_in = edge_idx.copy()

    connect_out = np.zeros(n_connect)
    connect_in = np.zeros(n_connect)

    connect_out[edge_out] = 1
    connect_in[edge_in] = 1

    a_mat = np.block(
        [
            [com_1, connect_out.reshape((half_nodes, -1))],
            [connect_in.reshape((half_nodes, -1)).T, com_2],
        ]
    )
    return a_mat


def toy_random_seeded(
    n_nodes,
    seed,
    out_prob=0.5,
    edge_prob=0.5,
    directed=False,
    con_prob=None,
    verbose=False,
):
    half_nodes = int(n_nodes / 2)

    seeds = [seed + i for i in range(4)]
    com_1 = random_graph(
        n_nodes=half_nodes, edge_prob=edge_prob, directed=directed, seed=seeds[0]
    )
    com_2 = random_graph(
        n_nodes=half_nodes, edge_prob=edge_prob, directed=directed, seed=seeds[1]
    )

    # connect_com = np.zeros(half_nodes**2)
    if con_prob is None:
        con_prob = edge_prob / 2
    n_connect = half_nodes**2
    np.random.seed(seeds[2])
    edge_idx = np.random.choice(n_connect, int(n_connect * con_prob), replace=False)
    if verbose:
        print(f"There will be {len(edge_idx)} connecting edges")

    if directed:
        np.random.seed(seeds[3])
        chosen_out_edges = np.random.choice(
            edge_idx,
            int(len(edge_idx) * out_prob),
            replace=False,
            # len(edge_idx), int(len(edge_idx) * out_prob), replace=False
        )
        edge_out_idx = np.array([i in chosen_out_edges for i in edge_idx])
        edge_out = edge_idx[edge_out_idx]
        edge_in = edge_idx[~edge_out_idx]
    else:
        edge_out = edge_idx.copy()
        edge_in = edge_idx.copy()

    connect_out = np.zeros(n_connect)
    connect_in = np.zeros(n_connect)

    connect_out[edge_out] = 1
    connect_in[edge_in] = 1

    if verbose:
        print(f"With {connect_out.sum()} out edges and {connect_in.sum()} in edges.")

    a_mat = np.block(
        [
            [com_1, connect_out.reshape((half_nodes, -1))],
            [connect_in.reshape((half_nodes, -1)).T, com_2],
        ]
    )

    if verbose:
        print(
            f"Average degrees are in: {a_mat.sum(axis=0).mean():1.3f}"
            f", out: {a_mat.sum(axis=1).mean():1.3f}"
        )
    return a_mat


def toy_n_communities(
    nodes_per_com: int,
    n_com: int = 3,
    com_density: float = 0.5,
    connect_density: list[float] = 0.5,
    connect_out_prob: list[float] = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:

    if seed is None:
        seed = int(np.random.normal(10000, 100))

    if isinstance(connect_density, (int, float)):
        connect_density = [connect_density] * (n_com * (n_com - 1) // 2)

    if isinstance(connect_out_prob, (int, float)):
        connect_out_prob = [connect_out_prob] * (n_com * (n_com - 1) // 2)

    n_connectors = n_com * (n_com - 1) // 2

    communities = [
        random_graph(
            nodes_per_com, edge_prob=com_density, directed=True, seed=seed + seed_i
        )
        for seed_i in range(n_com)
    ]

    if isinstance(connect_out_prob, (int, float)):
        connect_out_prob = [connect_out_prob] * n_connectors

    if isinstance(connect_density, (int, float)):
        connect_density = [connect_density] * n_connectors

    connectors = [
        random_connector(
            nodes_per_com,
            edge_prob=p_edge,
            directed=True,
            out_prob=p_out,
            seed=seed + seed_i,
        )
        for seed_i, (p_out, p_edge) in enumerate(zip(connect_out_prob, connect_density))
    ]

    if n_com == 3:
        adj = np.block(
            [
                [communities[0], connectors[0][0], connectors[1][0]],
                [connectors[0][1], communities[1], connectors[2][0]],
                [connectors[1][1], connectors[2][1], communities[2]],
            ]
        )
    if n_com == 4:
        adj = np.block(
            [
                [communities[0], connectors[0][0], connectors[1][0], connectors[2][0]],
                [connectors[0][1], communities[1], connectors[3][0], connectors[4][0]],
                [connectors[1][1], connectors[3][1], communities[2], connectors[5][0]],
                [connectors[2][1], connectors[4][1], connectors[5][1], communities[3]],
            ]
        )

    return adj
