from typing import List

import numpy as np
from dtaidistance.clustering import KMeans, KMedoids
from dtaidistance.dtw_barycenter import dba
from dtaidistance import dtw_ndim
from sklearn.metrics import silhouette_score


# class Implet:
#     def __init__(self, sample_id, implet_vals, implet_attr, score, start_loc, end_loc):
#         self.sample_id = sample_id
#         self.implet_vals = implet_vals
#         self.implet_attr = implet_attr
#         self.score = score
#         self.start_loc = start_loc
#         self.end_loc = end_loc
#
#     def __repr__(self):
#         s = 'Implet('
#         s += ', '.join([f'{k}={self.__dict__[k]}'
#                         for k in ['sample_id', 'start_loc', 'end_loc', 'score']])
#         s += ')'
#         return s


def _normalize_implets(implets: List[np.ndarray]):
    """
    Normalize each channel of the implets independently.
    The goal is to let features and attributions have roughly the same weight
    when computing distances.

    :param implets: list of arrays with shape (seq_len,) or (seq_len, 2).
    seq_len can be different for each entry.
    :return normalized_implets: normalized implets
    :return norm_params: tuple of (feature_mean, feature_std, attr_mean,
    attr_std)
    """
    if len(implets[0].shape) == 1:  # 1D
        features = np.concatenate([imp for imp in implets])
        feature_mean = np.mean(features)
        feature_std = np.std(features)

        normalized_implets = []
        for i in range(len(implets)):
            normalized_imp = (implets[i] - feature_mean) / feature_std
            normalized_implets.append(normalized_imp)

        return normalized_implets, (feature_mean, feature_std, None, None)

    else:  # 2D
        features = np.concatenate([imp[:, 0] for imp in implets])
        feature_mean = np.mean(features)
        feature_std = np.std(features)

        attr = np.concatenate([imp[:, 1] for imp in implets])
        attr_mean = np.mean(attr)
        attr_std = np.std(attr)

        normalized_implets = []
        for i in range(len(implets)):
            normalized_imp = np.zeros_like(implets[i])
            normalized_imp[:, 0] = (implets[i][:, 0] - feature_mean) / feature_std
            normalized_imp[:, 1] = (implets[i][:, 1] - attr_mean) / attr_std
            normalized_implets.append(normalized_imp)

        return normalized_implets, (feature_mean, feature_std, attr_mean, attr_std)


def silhouette_score_dtw(implets, labels):
    """
    Compute silhouette score for variable-length time series using DTW.

    Parameters:
    - implets: list of arrays, where each array is a time series.
    - labels: array-like, cluster labels for each time series.

    Returns:
    - float: silhouette score for the dataset.
    """
    n = len(implets)
    unique_labels = np.unique(labels)
    silhouette_scores = []

    implets, _ = _normalize_implets(implets)

    for i in range(n):
        # Compute a(i): mean DTW distance to other points in the same cluster
        same_cluster = [j for j in range(n) if labels[j] == labels[i] and i != j]
        if same_cluster:
            a_i = np.mean([dtw_ndim.distance(implets[i], implets[j]) for j in same_cluster])
        else:
            a_i = 0  # No other points in the same cluster

        # Compute b(i): minimum mean DTW distance to points in other clusters
        b_i = float('inf')
        for label in unique_labels:
            if label != labels[i]:
                other_cluster = [j for j in range(n) if labels[j] == label]
                if other_cluster:
                    b_i_cluster = np.mean([dtw_ndim.distance(implets[i], implets[j]) for j in other_cluster])
                    b_i = min(b_i, b_i_cluster)

        # Compute silhouette score for this point
        if len(same_cluster) > 0 and b_i != float('inf'):
            silhouette_scores.append((b_i - a_i) / max(a_i, b_i))

    # Return the average silhouette score
    return np.mean(silhouette_scores)


def max_score_subsequence(arr, left, lamb, threshold, kmin=None, kmax=None):
    n = len(arr)

    if kmin is None:
        kmin = 3
    if kmax is None:
        kmax = n // 2

    max_score = float('-inf')
    best_start, best_end = -1, -1
    current_sum = 0

    if arr[left] < threshold:
        return None, max_score, best_start, best_end

    for right in range(left + kmin - 1, min(left + kmax, n)):
        current_sum += arr[right]
        length = right - left + 1
        current_score = (current_sum / length) + lamb * length
        if length == 0:
            print(len(arr), right, left, kmin)
        if current_score > max_score and (current_sum / length) > threshold:
            max_score = current_score
            best_start, best_end = left, right
        # while left <= right and (current_sum / length + lamb*length) < max_score:
        #    current_sum -= arr[left]
        #    left += 1
    return arr[best_start:best_end + 1], max_score, best_start, best_end


def implet_extractor(train_x, train_y, attr, target_class=None, lamb=0.1, is_global_threshold=False, thresh_factor=1,
                     kmin=None, is_attr_abs=True):
    """
    extract implets from a dataset with a computed threshold. This method iterate through instances in datasets and put them into
    max_score_subsequence to find the subseuqnece that with the largest subsequence with each starting points.
    The threshold can be set globally or locally.
    :param train_x: time series instance
    :param train_y: time series labels
    :param attr: the attribution of time series instance
    :param target_class: if we only select part of data and extract labels
    :param lamb: lambda that used for max_score_subsequence
    :param is_attr_abs: control if using the absolute value of attribution
    :return: a dictionary [instance_number, the subsequence, the corresponding attribution,
                            sum of attribution, starting position, ending position]

    """
    implets = []
    num = len(train_x)
    global_threshold = None
    if is_global_threshold:
        if is_attr_abs:
            avg = np.mean(np.abs(attr))
            std = np.std(np.abs(attr))
        else:
            avg = np.mean(attr)
            std = np.std(attr)
        # avg = np.mean(np.abs(attr))
        # std = np.std(np.abs(attr))
        global_threshold = avg + 1 * std

    for i in range(num):
        inst = train_x[i].flatten()
        if target_class is not None and train_y[i] != target_class:
            continue
        starting = 0
        # abs_attr = np.abs(attr[i].flatten())
        input_attr = np.abs(attr[i].flatten()) if is_attr_abs else attr[i].flatten()

        avg = np.mean(input_attr)
        std = np.std(input_attr)
        threshold = global_threshold if is_global_threshold else avg + 1 * std
        # threshold = avg + thresh_factor * std
        # print(threshold)
        # threshold = avg + 0.6 * std
        while starting < len(input_attr):
            sub_attr, max_score, best_start, best_end = max_score_subsequence(arr=input_attr, left=starting, lamb=lamb,
                                                                              threshold=threshold, kmin=kmin)
            if best_end != -1:
                implets.append(
                    [i, inst[best_start:best_end + 1], attr[i, best_start:best_end + 1], max_score, best_start,
                     best_end])
                # implets.append(Implet(i,
                #                       inst[best_start:best_end + 1],
                #                       attr[i, best_start:best_end + 1],
                #                       float(max_score),
                #                       best_start, best_end))
                starting = best_end + 1
            else:
                starting += 1
    return implets


def implet_cluster(implets: List[np.ndarray], k: int):
    """
    Apply clustering on implets.
    If implets is higher dimensional (e.g. the first dim is features, the
    dim is importance), apply clustering based on dependent multi-dim DTW.
    :return cluster_indices: the indices of implets in each cluster
    :return centroids: centroids of each cluster
    """
    kmeans = KMeans(k=k, show_progress=False)
    # kmedoids = KMedoids(k=k, dists_fun=dtw_ndim.distance_matrix_fast, dists_options=dict())

    # normalization
    implets, (feature_mean, feature_std, attr_mean, attr_std) = _normalize_implets(implets)

    cluster_indices, _ = kmeans.fit(implets, use_parallel=False)
    # cluster_indices = kmedoids.fit(implets)
    # print(cluster_indices)

    # medoid_ids = list(cluster_indices.keys())
    # centroids = [implets[i] for i in medoid_ids]
    # _cluster_indices = {i: cluster_indices[medoid_ids[i]] for i in range(len(medoid_ids))}

    centroids = []
    for j in range(k):
        centroid = dba([implets[i] for i in cluster_indices[j]], None)
        centroids.append(centroid)

    # print(medoid_ids)

    for j in range(k):
        # reverse normalization
        if len(centroids[j].shape) == 1:
            centroids[j] = centroids[j] * feature_std + feature_mean
        else:
            centroids[j][:, 0] = centroids[j][:, 0] * feature_std + feature_mean
            centroids[j][:, 1] = centroids[j][:, 1] * attr_std + attr_mean

    return cluster_indices, centroids


def implet_cluster_auto(implets: List[np.ndarray], ks=None):
    """
    Apply clustering on implets and automatically choose k.
    :param implets:
    :param ks: if None, then test from 2 to 10.
    :return k:
    :return cluster_indices:
    :return centroids:
    """

    if ks is None:
        ks = range(1, min(5, len(implets)))

    best_silhouette = -1
    best_k = None
    best_indices = None
    best_centroids = None

    for k in ks:
        _indices, _centroids = implet_cluster(implets, k)
        labels = np.zeros(len(implets))
        for j, cluster in enumerate(_indices):
            for i in _indices[j]:
                labels[i] = j

        silhouette = silhouette_score_dtw(implets, labels)
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k
            best_indices = _indices
            best_centroids = _centroids

    return best_k, best_indices, best_centroids
