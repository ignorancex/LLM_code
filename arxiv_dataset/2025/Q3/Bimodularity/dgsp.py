# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score

from graph_examples import toy_random


# Some directed modularity tools
def configuration_null(a_mat: np.ndarray, null_model: str = "outin"):
    k_in = a_mat.sum(axis=0).reshape((1, -1))
    k_out = a_mat.sum(axis=1).reshape((1, -1))

    if null_model == "in":
        z = k_in.T @ k_in
    elif null_model == "out":
        z = k_out.T @ k_out
    elif null_model == "inout":
        z = k_in.T @ k_out
    elif null_model == "outin":
        z = k_out.T @ k_in
    elif null_model == "avg":
        z = (k_in.T @ k_in + k_out.T @ k_out) / 2
    elif null_model == "send":
        z = send_receive_probability(a_mat)[0]
    elif null_model == "receive":
        z = send_receive_probability(a_mat)[1]

    return z / a_mat.sum()


def send_receive_probability(adj: np.ndarray):
    n_edges = adj.sum()

    in_deg = np.atleast_2d(adj.sum(axis=0))
    out_deg = np.atleast_2d(adj.sum(axis=1))

    in_deg_squared = (in_deg**2).sum()
    out_deg_squared = (out_deg**2).sum()

    send_prob = np.outer(out_deg, out_deg) * in_deg_squared / n_edges**2
    receive_prob = np.outer(in_deg, in_deg) * out_deg_squared / n_edges**2

    return send_prob, receive_prob


def modularity_matrix(a_mat: np.ndarray, null_model: str = "outin"):

    z = configuration_null(a_mat, null_model)

    return a_mat - z


def sorted_SVD(matrix: np.ndarray, fix_negative: bool = False, sort_by_q: bool = False):

    U, S, Vh = np.linalg.svd(matrix, full_matrices=True)

    if fix_negative:
        for i, _ in enumerate(matrix):
            if U[:, i].T @ Vh[i] < 0:
                Vh[i] *= -1
                S[i] *= -1

    sort_id = np.flip(np.argsort(S))
    if sort_by_q:
        q_s = S * np.diag(Vh @ U)
        sort_id = np.flip(np.argsort(q_s))

    S = S[sort_id]
    U = U[:, sort_id]
    Vh = Vh[sort_id]

    return U, S, Vh


# Community detection part


def edge_bicommunities(
    adjacency,
    U,
    V,
    n_components,
    method="partition",
    n_kmeans=10,
    verbose=False,
    scale_S=None,
    assign_only=False,
    **kwargs,
) -> tuple:

    n_nodes = adjacency.shape[0]

    if scale_S is None:
        scale_S = np.ones(n_components)

    # u_features = U[:, :n_components] * np.sqrt(scale_S)
    # v_features = V[:, :n_components] * np.sqrt(scale_S)

    u_features = U[:, :n_components] * scale_S
    v_features = V[:, :n_components] * scale_S

    if method in ["partition", "sign"]:
        u_features = np.sign(u_features).astype(int)
        v_features = np.sign(v_features).astype(int)

    edge_out = np.array([u_features] * n_nodes).T
    edge_in = np.array([v_features] * n_nodes)
    edge_in = np.moveaxis(edge_in, -1, 0)

    # edge-based clustering
    edge_assignments = np.concatenate([edge_out, edge_in], axis=0)
    edge_assignments_vec = edge_assignments.reshape((2 * n_components, -1)).T

    edge_assignments_vec = edge_assignments_vec[(adjacency != 0).reshape(-1)]

    if assign_only:
        return edge_assignments_vec

    if method in ["partition", "sign"]:
        clusters = np.unique(edge_assignments_vec, axis=0)
        n_clusters = clusters.shape[0]
        if verbose:
            print(f"Found {n_clusters} clusters !")

        cluster2num = {tuple(c): i + 1 for i, c in enumerate(clusters)}

        edge_clusters = np.array([cluster2num[tuple(c)] for c in edge_assignments_vec])

    elif method == "kmeans":
        if n_kmeans is None:
            n_kmeans = get_best_k(edge_assignments_vec, verbose=verbose, **kwargs)

        kmeans = KMeans(n_clusters=n_kmeans, random_state=0, n_init="auto").fit(
            edge_assignments_vec
        )
        edge_clusters = kmeans.labels_ + 1

        n_clusters = edge_clusters.max()
        if verbose:
            print(f"Found {n_clusters} clusters !")
    else:
        raise ValueError(
            "Method not recognized (possible values: partition, sign, kmeans)"
        )

    edge_clusters_mat = np.zeros((n_nodes, n_nodes), dtype=int)
    edge_clusters_mat[adjacency != 0] = edge_clusters

    return edge_clusters, edge_clusters_mat


def get_best_k(X, max_k=10, verbose=False):
    print(f"Running silhouette analysis for k = 2 to {max_k} ...")
    n_clusters = np.arange(2, max_k)
    silhouette = np.zeros(n_clusters.shape[0])

    for i, n in enumerate(n_clusters):
        kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto").fit(X)
        silhouette[i] = silhouette_score(X, kmeans.labels_)

        if verbose:
            print(f"Silhouette score for K={n} is : {silhouette[i]:1.2f}")

    print(
        f"Best average silhouette_score is : {np.max(silhouette):1.2f} for K={n_clusters[np.argmax(silhouette)]}"
    )
    return n_clusters[np.argmax(silhouette)]


def get_node_clusters(edge_clusters, edge_clusters_mat, method="bimod", scale=True):
    n_nodes = edge_clusters_mat.shape[0]
    n_clusters = np.max(edge_clusters)

    if method == "probability":
        # Aggregate edges to nodes using cluster probability (number of edges)
        n_per_cluster = np.zeros((n_nodes, n_clusters))
        for cluster_id in np.arange(1, np.max(edge_clusters_mat) + 1):
            n_per_cluster[:, cluster_id - 1] = np.sum(
                edge_clusters_mat == cluster_id, axis=1
            )
            n_per_cluster[:, cluster_id - 1] += np.sum(
                edge_clusters_mat == cluster_id, axis=1
            )

        cluster_prob = n_per_cluster / n_per_cluster.sum(axis=1)[:, None]

        cluster_maxprob = np.argmax(cluster_prob, axis=1) + 1

        return cluster_maxprob, cluster_prob
    if "bimod" in method:
        sending_communities = np.zeros((n_clusters, n_nodes))
        receiving_communities = np.zeros((n_clusters, n_nodes))

        for cluster_id in np.arange(1, np.max(edge_clusters_mat) + 1):
            sending_communities[cluster_id - 1] = np.sum(
                edge_clusters_mat == cluster_id, axis=1
            )
            receiving_communities[cluster_id - 1] = np.sum(
                edge_clusters_mat == cluster_id, axis=0
            )

        if scale:
            sending_communities = np.nan_to_num(
                sending_communities / np.sum(edge_clusters_mat > 0, axis=1),
                posinf=0,
                neginf=0,
            )
            receiving_communities = np.nan_to_num(
                receiving_communities / np.sum(edge_clusters_mat > 0, axis=0),
                posinf=0,
                neginf=0,
            )

        return sending_communities, receiving_communities


def bimod_index_nodes(adjacency, send_com, receive_com, scale=False):
    n_clusters = len(send_com)

    null_model = configuration_null(adjacency, null_model="outin")

    bimod_indices = np.zeros(n_clusters)

    for cluster_id in np.arange(n_clusters):
        send_fltr = send_com[cluster_id] > 0
        receive_fltr = receive_com[cluster_id] > 0

        adj_contrib = adjacency[send_fltr][:, receive_fltr]
        null_contrib = null_model[send_fltr][:, receive_fltr]

        bimod_indices[cluster_id] = np.sum(adj_contrib - null_contrib)

        if scale:
            all_edges = np.sum(np.atleast_2d(send_fltr).T @ np.atleast_2d(receive_fltr))
            bimod_indices[cluster_id] /= all_edges
        # / np.sum(adj_contrib > 0)

    return bimod_indices
