import numpy as np
from torch.utils.data import Subset, ConcatDataset

from .server_sample import server_sample
from .step_partition import step_partition
from .dirichlet_partition import dirichlet_partition

from .stat import print_quantity_stat, print_label_distribution_stat


def create_partition(dataset, args, return_raw=False):

    num_clients = args.num_honest  # number of honest clients
    num_labels = args.num_labels
    partition_config = args.partition

    partition_idxs = partition(dataset, num_labels, num_clients, partition_config)

    print_label_distribution_stat(dataset, num_labels, partition_idxs, visualize=args.visualize, resize=0.2)

    train_client_sample_ids = {
        cid: {'train': sample_ids} for cid, sample_ids in partition_idxs.items()
    }

    server_sample_ids = server_sample(dataset, num_labels, args)

    if return_raw:

        return train_client_sample_ids, server_sample_ids, partition_idxs

    else:

        return train_client_sample_ids, server_sample_ids


def partition(dataset, num_labels, num_clients, partition_config):
    """
    Partition a dataset to several clients. However, there is no train-test split or sampling.
    """
    # parse partition method and parameters
    alg, *params = partition_config.split('_')

    # partition
    if alg == 'step':
        num_major = int(params[0])
        alpha = float(params[1])
        partition_idxs = step_partition(dataset, num_labels, num_clients, num_major, alpha)

    # elif alg == 'patho' or alg == 'fedavg':
    #     num_major = int(params[0])
    #     alpha = float('inf')
    #     partition_idxs = step_partition(dataset, num_labels, num_clients, num_major, alpha)

    elif alg == 'dir':
        alpha = float(params[0])
        partition_idxs = dirichlet_partition(dataset, num_labels, num_clients, alpha)

    elif alg == 'stratified':
        num_major = int(params[0])
        alpha = 1.0
        partition_idxs = step_partition(dataset, num_labels, num_clients, num_major, alpha)


    else:
        raise NotImplementedError('Unknown data partition algorithm. ')

    return partition_idxs
