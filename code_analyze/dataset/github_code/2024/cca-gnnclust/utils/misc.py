import numpy as np

def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec

def stop_iterating(
    current_l,
    total_l,
    early_stop,
    num_edges_add_this_level,
    num_edges_add_last_level,
    knn_k,
):
    # Stopping rule 1: run all levels
    if current_l == total_l - 1:
        return True
    # Stopping rule 2: no new edges
    if num_edges_add_this_level == 0:
        return True
    # Stopping rule 3: early stopping, two levels start to produce similar numbers of edges
    if (
        early_stop
        and float(num_edges_add_last_level) / num_edges_add_this_level
        < knn_k - 1
    ):
        return True
    return False
