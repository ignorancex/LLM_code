"""
Step Partition
"""

import numpy as np

from .utils import get_labels

import itertools


def step_partition(dataset, num_labels, num_clients, num_major, alpha):
    """
    :param dataset: Dataset
    :param num_clients: number of clients ()
    :param alpha: concentration score. Larger alpha -> more IID
    :return:
    """

    labels, idxs_by_label, num_samples_per_label = get_labels(dataset, num_labels)

    if alpha == float('inf'):
        matrix = np.zeros((num_clients, num_labels))
        alpha = 2
    else:
        matrix = np.ones((num_clients, num_labels))

    if num_clients == 100 and num_labels == 10:  # MNIST and CIFAR-10 experiments
        for cid, label_ids in enumerate(itertools.product(range(num_labels), repeat=num_major)):
            for label_id in label_ids:
                matrix[cid, label_id] += (alpha - 1)

    elif num_clients == 160 and num_labels == 4 and num_major == 2:  # AG News experiments
        for cid, label_ids in enumerate(itertools.product(range(num_labels), repeat=num_major)):
            for label_id in label_ids:
                matrix[cid * 10:(cid + 1) * 10, label_id] += (alpha - 1)


    elif num_clients == 16 and num_labels == 4 and num_major == 2:  # AG News experiments for ablation study
        for cid, label_ids in enumerate(itertools.product(range(num_labels), repeat=num_major)):
            for label_id in label_ids:
                matrix[cid, label_id] += (alpha - 1)

    else:
        raise NotImplementedError

    # normalizing matrix
    matrix = matrix / matrix.sum(axis=0)

    # cumulative matrix
    cumulate = matrix.cumsum(axis=0) * num_samples_per_label
    cumulate = (cumulate + 0.5).astype(int)  # round to integer
    cumulate = np.vstack([np.zeros((1, num_labels), dtype=int), cumulate])

    partition_idxs = dict()

    for cid in range(num_clients):
        idxs = []
        for label in range(num_labels):
            idxs.append(idxs_by_label[label][cumulate[cid, label]:cumulate[cid + 1, label]])

        partition_idxs[cid] = np.concatenate(idxs)

    return partition_idxs
