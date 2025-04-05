"""
Preprocess and partition a centralized dataset to clients in FL
"""
import os.path

import numpy as np

from dataset import create_dataset, shapes_out
from partition import create_partition
from utils import GloVe, pickle_save
from options import args_parser


def main(args):
    """
    :return: client_sample_id
    {cid: {'train': list of sample ids}}
    """
    config = {}
    if args.dataset in ['agnews', ]:
        embed = GloVe(root=args.data_dir)
        config['embed'] = embed

    if args.dataset == 'cifar10c':  # preprocessing dataset

        from corruption import make_cifar10_c

        # get dataset
        train_dataset, test_dataset = create_dataset('cifar10', args.data_dir, config)

        # get partition of datasets
        train_client_sample_id, server_sample_id, partition_idxs = create_partition(train_dataset, args,
                                                                                    return_raw=True)

        path = os.path.join(args.data_dir, 'boba', 'cifar10c_seed_' + str(args.partition_seed))
        cifar_c, labels, corruption = make_cifar10_c(partition_idxs, severity=args.severity, data_dir=args.data_dir)

        obj = {
            'X': cifar_c,
            'Y': labels,
            'corruption': corruption,
        }

        pickle_save(obj=obj, file=args.corruption_path, mode='wb')

    else:

        # get dataset
        train_dataset, test_dataset = create_dataset(args.dataset, args.data_dir, config)

        # get partition of datasets
        train_client_sample_id, server_sample_id = create_partition(train_dataset, args)

    # check correctness
    all_sample = set()
    for cid, datasets in train_client_sample_id.items():
        for part, idxs in datasets.items():
            all_sample = all_sample.union(set(idxs))

    print('Before partition:', len(train_dataset))
    print('After partition: ', len(all_sample))

    sample_id = (train_client_sample_id, server_sample_id)

    pickle_save(obj=sample_id, file=args.partition_path, mode='wb')


def set_seed(seed):
    np.random.seed(seed)


if __name__ == '__main__':
    args = args_parser()
    set_seed(args.partition_seed)
    main(args)
