from torch.utils.data import ConcatDataset, Subset, TensorDataset

from utils import pickle_load

from .torchvision_dataset import create_torchvision_dataset
from .AG_NEWS import AG_NEWS
from .Spambase import Spambase
from .NoisyDataset import NoisyDataset


def create_dataset(dataset_name, data_dir, config=None):
    """
    Create the dataset and its partition
    :param args:
    :return:
    """
    torchvision_dataset_names = [
        'mnist',
        'cifar10',
    ]

    if dataset_name in torchvision_dataset_names:
        datasets = create_torchvision_dataset(dataset_name=dataset_name, data_dir=data_dir, cache=True)

    elif dataset_name == 'agnews':
        train_dataset = AG_NEWS(root=data_dir, vocab=config['embed'].stoi, train=True, download=True)
        test_dataset = AG_NEWS(root=data_dir, vocab=config['embed'].stoi, train=False, download=True)
        datasets = (train_dataset, test_dataset)

    elif dataset_name == 'spambase':
        train_dataset = Spambase(root=data_dir, train=True, download=True)
        test_dataset = Spambase(root=data_dir, train=False, download=True)
        datasets = (train_dataset, test_dataset)

    else:
        raise NotImplementedError('Unknown dataset!')

    return datasets


def create_fed_dataset(args, config=None, central_test=True):
    dataset_name = args.dataset
    data_dir = args.data_dir
    partition_path = args.partition_path

    if central_test:
        if dataset_name in ['mnist', 'fmnist', 'cifar10', 'cifar100', 'coarse-cifar100', 'agnews', 'spambase']:
            train_dataset, test_dataset = create_dataset(dataset_name, data_dir, config)
            train_client_sample_id, server_sample_id = pickle_load(partition_path)

            train_datasets = {
                cid: {part: Subset(train_dataset, indices) for part, indices in sids.items()} for cid, sids in
                train_client_sample_id.items()
            }

            server_datasets = {part: Subset(train_dataset, indices) for part, indices in server_sample_id.items()}

        elif dataset_name == 'cifar10c':
            # test and server dataset
            temp_dataset, test_dataset = create_dataset('cifar10', data_dir, config)
            train_client_sample_id, server_sample_id = pickle_load(partition_path)

            server_datasets = {part: Subset(temp_dataset, indices) for part, indices in server_sample_id.items()}

            obj = pickle_load(args.corruption_path)
            X = obj['X']
            Y = obj['Y']

            train_dataset = TensorDataset(X, Y)

            train_datasets = {
                cid: {part: Subset(train_dataset, indices) for part, indices in sids.items()} for cid, sids in
                train_client_sample_id.items()
            }


        else:
            raise NotImplementedError

        if args.server_data_noise != 'none':
            server_datasets2 = {
                key: NoisyDataset(clean_dataset=value, noise=args.server_data_noise, severity=3, args=args)
                for key, value in server_datasets.items()
            }
            server_datasets = server_datasets2

        return train_datasets, server_datasets, test_dataset

    else:

        raise NotImplementedError
