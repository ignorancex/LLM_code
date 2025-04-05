import numpy as np

from .utils import get_labels
from .stat import print_label_distribution_stat


def server_sample(dataset, num_labels, args):

    num_samples_per_class = args.num_server_data_per_class

    labels, idxs_by_label, num_samples_per_label = get_labels(dataset, num_labels)

    dict_server = {}

    server_idxs = []

    if not args.biased_server_sample:
        for i in range(num_labels):
            idxs = idxs_by_label[i][:num_samples_per_class]
            dict_server[i] = idxs
            server_idxs.append(idxs)

    else:
        for i in range(num_labels // 2):
            idxs = idxs_by_label[i][:num_samples_per_class]
            dict_server[i] = idxs
            server_idxs.append(idxs)

        for i in range(num_labels // 2, num_labels):
            idxs = idxs_by_label[i][:(num_samples_per_class // 2)]
            dict_server[i] = idxs
            server_idxs.append(idxs)

    print_label_distribution_stat(dataset, num_labels, dict_server, visualize=False)

    dict_server['all'] = np.concatenate(server_idxs)

    return dict_server
