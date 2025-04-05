"""
Dirichlet Partition
https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py
"""

import numpy as np

from .utils import get_labels

import itertools


def dirichlet_partition(dataset, num_labels, num_clients, alpha):
    """
    :param dataset: Dataset
    :param num_clients: number of clients ()
    :param alpha: concentration score. Larger alpha -> more IID
    :return:
    """
    labels, idxs_by_label, num_samples_per_label = get_labels(dataset, num_labels)

    num_samples = len(dataset)
    base = num_samples // num_clients
    parts = np.full(num_clients, base)

    remainder = num_samples % num_clients
    parts[:remainder] += 1

    return client_inner_dirichlet_partition_faster(targets=labels, num_clients=num_clients, num_classes=num_labels,
                                                   dir_alpha=alpha, client_sample_nums=parts, verbose=False)


def client_inner_dirichlet_partition_faster(targets, num_clients, num_classes, dir_alpha,
                                            client_sample_nums, verbose=True):
    """Non-iid Dirichlet partition.

    The method is from The method is from paper `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_.
    This function can be used by given specific sample number for all clients ``client_sample_nums``.
    It's different from :func:`hetero_dir_partition`.

    Args:
        targets (list or numpy.ndarray): Sample targets.
        num_clients (int): Number of clients for partition.
        num_classes (int): Number of classes in samples.
        dir_alpha (float): Parameter alpha for Dirichlet distribution.
        client_sample_nums (numpy.ndarray): A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    class_priors = np.random.dirichlet(alpha=[dir_alpha] * num_classes,
                                       size=num_clients)
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]

    client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in
                      range(num_clients)]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        # If current node is full resample a client
        if verbose:
            print('Remaining Data: %d' % np.sum(client_sample_nums))
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if no rest in current class samples
            if class_amount[curr_class] <= 0:
                # Exception handling: If the current class has no samples left, randomly select a non-zero class
                while True:
                    new_class = np.random.randint(num_classes)
                    if class_amount[new_class] > 0:
                        curr_class = new_class
                        break
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                idx_list[curr_class][class_amount[curr_class]]

            break

    client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
    return client_dict
