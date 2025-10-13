import networkx as nx
import numpy as np


def decompose_preorder(adjacency):
    """
    Decompose a preorder into a clustering and the transitive reduction of the partial order on this clustering
    :param adjacency: n x n matrix where a_ij >= 0.5 indicates that ij is in the preorder
    :return: transitive reduction [nx.DiGraph] of partial order of clusters and clustering [list]
    """
    n = adjacency.shape[0]
    # compute clustering
    visited = np.zeros(n, dtype=bool)
    clustering = []
    for i in range(n):
        if visited[i]:
            continue
        mask = np.logical_and(adjacency[i] >= 0.5, adjacency[:, i] >= 0.5)
        visited[mask] = True
        clustering.append(list(np.where(mask)[0]))

    reps = [comp[0] for comp in clustering]  # get one representative of each cluster
    adjacency = adjacency[reps][:, reps]  # reduce adjacency to representatives
    adjacency = adjacency >= 0.5  # get partial order on representatives
    adjacency[np.diag_indices_from(adjacency)] = False
    g = nx.DiGraph(adjacency)  # convert adjacency matrix to directed graph
    assert nx.is_directed_acyclic_graph(g)
    transitive_reduction_graph = nx.transitive_reduction(g)  # compute transitive reduction of partial order
    return transitive_reduction_graph, clustering
