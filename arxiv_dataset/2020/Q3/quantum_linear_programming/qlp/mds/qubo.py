"""Tools to generate a Minimum Dominating Set QUBO from a graph
"""
from typing import Set, Tuple, List, Optional
from scipy.sparse import dok_matrix, bmat, triu, tril
import numpy as np


def get_adjacency(graph: Set[Tuple[int]], directed: bool = False) -> dok_matrix:
    """This routine computes the adjecency matrix for a given graph.

    It assumes the graph is connected and nodes are labeled from 0...N-1.
    Self loops are forbidden.

    Arguments:
        graph: The graph. If the graph is directed, first entry points to second.
        directed: Whether or not this is a directed graph. If not directed, the graph
            is symmeterized.

    """
    if not isinstance(graph, set):
        raise TypeError("Graph must be a set of tuples")

    nodes = set()
    adj = dict()

    for v1, v2 in graph:
        if v1 == v2:
            raise KeyError(f"Self loops are not allowed (v={v1}).")

        nodes.add(v1)
        nodes.add(v2)
        adj[(v1, v2)] = 1

    nodes = sorted(list(nodes))
    n_nodes = len(nodes)

    # if nodes != list(range(n_nodes)):
    #    raise KeyError("Nodes must be from 0 ... n_nodes")

    adjacency = dok_matrix((n_nodes, n_nodes), dtype=int)
    for (v1, v2), val in adj.items():
        adjacency[(v1, v2)] = val
        if not directed:
            adjacency[(v2, v1)] = val

    if not all([val == 1 for val in adjacency.values()]):
        raise KeyError("Double entries in graph")

    return adjacency


def get_bitmap(n_neigbors: List[int]) -> dok_matrix:
    r"""Computes the map of slack-bits to slack integers.

    This matrix B maps the bit vector b to slack variables s.
    Each s_i has max(s_i) >= N_i where N_i is the number of
    neigbors of vertex i.
    The bit vector b = (b_00, b_01, b_02, ... , b_11, b_12, ...)
    with b_ij will be mapped onto s_iaccording to
    s_i = sum_j 2**j b_ij.
    Thus the size of the returned matrix scales as
    ~ sum_i log2(N_i)

    Arguments:
        n_neigbors: The number of neighbors for each vertex.
    """
    n_nodes = len(n_neigbors)

    # Figure out how many bits we need for each neighborhood
    n_bits = np.array(
        [np.floor(np.log2(nn) + 1).astype(int) if nn > 0 else 0 for nn in n_neigbors]
    )
    bitmap = dok_matrix((n_nodes, int(n_bits.sum())), dtype=int)

    acc = 0
    for v, n_bit in enumerate(n_bits):
        for n in range(n_bit):
            bitmap[(v, acc)] = 2 ** n
            acc += 1

    return bitmap


def get_mds_qubo(
    graph: Set[Tuple[int]],
    directed: bool = False,
    triangularize: bool = False,
    penalty: Optional[int] = None,
    dtype: str = "i",
) -> dok_matrix:
    """This routine computes Minimum Dominating Set QUBO for a given graph.

    It assumes the graph is connected and nodes are labeled from 0...N-1.
    Self loops are forbidden.

    Arguments:
        graph: The graph. If the graph is directed, first entry points to second.
        directed: Whether or not this is a directed graph.
        triangularize: Put lower triangular entries in the upper diagonal.
        penalty: Energy penalty for violating constraints. Defaults to `n_nodes + 1`.
        dtype: Data type of qubo arrays. Defaults to "i" (integer). Float is "d".
    """

    ## This is J
    adjacency = get_adjacency(graph, directed=directed).astype(dtype)

    ## Id in x-space
    one = dok_matrix(adjacency.shape, dtype=dtype)
    for n in range(adjacency.shape[0]):
        one[(n, n)] = 1

    ## |J| (note that this is different than the n_neighbors)
    adjacency_sum = adjacency.sum(axis=0).flatten().tolist()[0]

    ## diag(|N|)
    diag_neighbor = dok_matrix(adjacency.shape, dtype=dtype)
    for n, el in enumerate(adjacency_sum):
        diag_neighbor[(n, n)] = el

    ## T (note that this is different than the adjacency_sum)
    n_neigbors = adjacency.sum(axis=1).flatten().tolist()[0]
    bitmap = get_bitmap(n_neigbors).astype(dtype)
    n_bits = bitmap.shape[1]

    ## diag(|T|)
    diag_bitmap = dok_matrix((n_bits, n_bits), dtype=dtype)
    for n, el in enumerate(bitmap.sum(axis=0).tolist()[0]):
        diag_bitmap[(n, n)] = el

    ## Compute QUBO components
    q_xx = adjacency.T @ adjacency + adjacency + adjacency.T - 2 * diag_neighbor - one
    q_xs = -2 * (one + adjacency.T) @ bitmap
    q_ss = bitmap.T @ bitmap + 2 * diag_bitmap

    ## Multiply by penalty factor
    penalty = penalty or len(n_neigbors) + 1
    q_xx *= penalty
    q_xs *= penalty
    q_ss *= penalty

    ## Add in minimization condition
    q_xx += one

    ## Construt QUBO
    q = bmat([[q_xx, q_xs], [None, q_ss]])

    if triangularize:
        q = triu(q) + tril(q, -1).T
    return q.todok()


def main(col_wrap: int = 4):  # pylint: disable=R0914
    """Generates a random graph

    Arguments:
        col_wrap: Plot at most graphs `col_wrap` in one row.
    """
    from argparse import ArgumentParser
    import matplotlib.pylab as plt
    from qlp.mds.graph_tools import generate_graph, get_plot_mpl
    from qlp.mds.solver import classical_search

    parser = ArgumentParser(
        "Generate random graph,"
        " compute QUBO of Minimum Dominating Set problem"
        " and plot minimum solutions."
    )
    parser.add_argument("-n", "--n-nodes", type=int, help="Number of nodes.", default=5)
    parser.add_argument("-e", "--n-edges", type=int, help="Number of edges.", default=5)
    parser.add_argument(
        "-m", "--n-edge-max", type=int, help="Max number of edges per node.", default=3
    )
    args = parser.parse_args()

    n_nodes = args.n_nodes
    n_edges = args.n_edges
    n_edge_max = args.n_edge_max

    test_graph = generate_graph(n_nodes, n_edges, n_edge_max=n_edge_max)
    qubo = get_mds_qubo(test_graph)
    e_min, solutions = classical_search(qubo, n_nodes=n_nodes)

    ## Note that the energy offset (constant part of penalty not in QUBO) is of size
    ### (n_nodes + 1) * n_nodes where the first term is the penalty and the second one
    ### The number of constraints.
    e_min += (n_nodes + 1) * n_nodes

    ## Check if constrained is violated, e.g., slacks contribute to energy
    if any([len(sol) != e_min for sol in solutions]):
        raise ValueError(
            "Constraint not fulfilled for graph:"
            f"\n{test_graph}"
            "\nand solutions:"
            f"\n{solutions}"
        )

    # Plotting
    n_sols = len(solutions)
    n_rows = n_sols // col_wrap + (1 if n_sols % col_wrap != 0 else 0)
    n_cols = col_wrap if col_wrap < n_sols else n_sols

    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, squeeze=False)
    for ax in axs.flatten():
        ax.set_visible(False)
    for sol, ax in zip(solutions, axs.flatten()):
        ax.set_visible(True)
        get_plot_mpl(test_graph, color_nodes=sol, ax=ax)

    fig.suptitle(rf"$\gamma(G) \leq {e_min}$")
    plt.show()


def QUBO_to_Ising(Q):
    """Maps QUBO to Ising model
    """
    q = np.diagonal(Q)
    QD = np.copy(Q)
    for i in range(len(QD)):
        QD[i, i] = 0.0
    QQ = np.copy(QD + np.transpose(QD))
    J = np.triu(QQ) / 4.0
    uno = np.ones(len(QQ))
    h = q / 2 + np.dot(QQ, uno) / 4
    g = np.dot(uno, np.dot(QD, uno)) / 4.0 + np.dot(q, uno) / 2.0
    return (J, h, g)


if __name__ == "__main__":
    main()
