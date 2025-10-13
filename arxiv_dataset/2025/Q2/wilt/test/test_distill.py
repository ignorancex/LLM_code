import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from typing import Optional

from torch_geometric.datasets import TUDataset  # type: ignore

from distill import PairSampler, train_gd  # type: ignore
from path import DATA_DIR  # type: ignore
from tree import WeisfeilerLemanLabelingTree  # type: ignore


@pytest.mark.parametrize(
    "dataset_name, batch_size, train",
    [
        ("MUTAG", 10, True),
        ("MUTAG", 20, False),
        ("Mutagenicity", 10, True),
        ("Mutagenicity", 20, False),
        ("NCI1", 10, True),
        ("NCI1", 20, False),
    ],
)
def test_PairSampler(dataset_name: str, batch_size: int, train: bool):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    sampler = PairSampler(data, batch_size, torch.ones((len(data), len(data))), train)
    for left_indices, right_indices, distances in sampler:
        assert (
            left_indices.shape
            == right_indices.shape
            == distances.shape
            == (batch_size,)
        )
        break
    assert sampler.all_pairs.shape == (len(data) * (len(data) - 1) // 2, 2)


@pytest.mark.parametrize(
    "dataset_name, depth, normalize, seed, l1coeff, batch_size, n_epochs, lr, save_interval",
    [
        ("MUTAG", 2, "size", 0, 0.01, 32, 1, 0.01, 1),
        ("MUTAG", 3, "dummy", 42, 0, 16, 2, 0.01, 1),
    ],
)
def test_train_gd(
    tmpdir,
    dataset_name: str,
    depth: int,
    normalize: str,
    seed: int,
    l1coeff: float,
    batch_size: int,
    n_epochs: int,
    lr: float,
    save_interval: int,
):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth, normalize)
    train_gd(
        data,
        tree,
        seed,
        str(tmpdir),
        l1coeff,
        batch_size,
        n_epochs,
        lr,
        save_interval,
        torch.ones(len(data), len(data)),
    )
