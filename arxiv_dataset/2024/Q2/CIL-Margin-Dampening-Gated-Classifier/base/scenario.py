from functools import lru_cache
from itertools import permutations
from typing import Optional

import numpy as np
import torch
from avalanche.benchmarks import nc_benchmark

from avalanche.benchmarks.classic.ccifar10 import _default_cifar10_train_transform, _default_cifar10_eval_transform

from avalanche.benchmarks.classic.ccifar100 import _default_cifar100_train_transform, _default_cifar100_eval_transform

from avalanche.benchmarks.classic.ctiny_imagenet import  _get_tiny_imagenet_dataset

from avalanche.benchmarks.classic.ctiny_imagenet import \
    _default_train_transform as _default_tiny_imagenet_train_transform
from avalanche.benchmarks.classic.ctiny_imagenet import \
    _default_eval_transform as _default_tiny_imagenet_eval_transform
from avalanche.benchmarks.datasets.external_datasets import get_cifar10_dataset, \
    get_cifar100_dataset


def get_permutation(n_classes):
    @lru_cache(maxsize=None)
    def rule_asc(n):
        a = [0 for i in range(n + 1)]
        k = 1
        a[1] = n
        while k != 0:
            x = a[k - 1] + 1
            y = a[k] - 1
            k -= 1
            while x <= y:
                a[k] = x
                y -= x
                k += 1
            a[k] = x + y
            yield a[:k + 1]

    divisions = list(filter(lambda x: not 1 in x and len(x) > 1,
                            rule_asc(n_classes)))

    all_divisions = []
    for d in divisions:
        all_divisions.extend(set(permutations(d)))

    selected = all_divisions[np.random.randint(0, len((all_divisions)))]

    return {i: v for i, v in enumerate(selected)}


def get_dataset_by_name(name: str, root: str = None):
    name = name.lower()

    if name == 'cifar10':
        train_set, test_set = get_cifar10_dataset(root)
        train_t = _default_cifar10_train_transform
        test_t = _default_cifar10_eval_transform
    elif name == 'cifar100':
        train_set, test_set = get_cifar100_dataset(root)
        train_t = _default_cifar100_train_transform
        test_t = _default_cifar100_eval_transform
    elif name == 'tinyimagenet':
        train_set, test_set = _get_tiny_imagenet_dataset(root)
        train_t = _default_tiny_imagenet_train_transform
        test_t = _default_tiny_imagenet_eval_transform
    else:
        return None

    return train_set, test_set, train_t, test_t


def get_dataset_nc_scenario(name: str, n_tasks: int, til: bool,
                            shuffle: bool = True, seed: Optional[int] = None,
                            force_sit=False, method_name=None, dev_split=None,
                            permuted_dataset: bool = False):

    name = name.lower()

    r = get_dataset_by_name(name)

    if r is None:
        assert False, f'Dataset {name} not found.'

    train_split, test_split, train_t, test_t = r

    if permuted_dataset:
        rp = get_permutation(len(set(train_split.targets)))
        n_tasks = len(rp)
    else:
        rp = None

    if dev_split is not None:
        idx = np.arange(len(train_split))
        np.random.RandomState(0).shuffle(idx)

        if isinstance(dev_split, int):
            dev_i = dev_split
        else:
            dev_i = int(len(idx) * dev_split)

        dev_idx = idx[:dev_i]
        train_idx = idx[dev_i:]

        test_split = torch.utils.data.Subset(train_split, dev_idx)
        train_split = torch.utils.data.Subset(train_split, train_idx)

    if method_name == 'cope':
        return nc_benchmark(
            train_dataset=train_split,
            test_dataset=test_split,
            n_experiences=n_tasks,
            task_labels=True,
            seed=seed,
            fixed_class_order=None,
            per_exp_classes=rp,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=False,
            class_ids_from_zero_from_first_exp=True,
            train_transform=train_t,
            eval_transform=test_t)

    if til and not force_sit:
        return nc_benchmark(
            train_dataset=train_split,
            test_dataset=test_split,
            n_experiences=n_tasks,
            task_labels=True,
            seed=seed,
            fixed_class_order=None,
            per_exp_classes=rp,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_t,
            eval_transform=test_t)
    else:
        return nc_benchmark(
            train_dataset=train_split,
            test_dataset=test_split,
            n_experiences=n_tasks,
            task_labels=False,
            seed=seed,
            class_ids_from_zero_from_first_exp=True,
            fixed_class_order=None,
            per_exp_classes=rp,
            shuffle=shuffle,
            train_transform=train_t,
            eval_transform=test_t)