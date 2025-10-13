import torch
import math
import numpy as np
from functools import partial
from enum import Enum
from collections import Counter
from torch_geometric.utils import homophily, to_networkx
from networkx import pagerank, diameter, connected_components
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Reddit2
from torch_geometric.nn.functional import gini
from torch_geometric.utils import to_undirected, is_undirected
from matplotlib import pyplot as plt

#disable cuda
#torch.cuda.is_available = lambda: False

device = "cuda" if torch.cuda.is_available() else "cpu"

datasets = {"Cora": Planetoid,
            "CiteSeer": Planetoid,
            "PubMed": Planetoid,
            "Reddit2": lambda name, **kwargs: Reddit2(**kwargs),
            "ogbn-arxiv": PygNodePropPredDataset}

n_class_estimations = {"Cora": 11,
                       "CiteSeer": 13,
                       "PubMed": 7,
                       "Reddit2": 50, # todo
                       "ogbn-arxiv": 50} # todo

def create_split(data, train_portion=0.0, val_portion=0.7, seed=None):
    """Splits the dataset into train, validation, and test

    :param data: Dataset to split
    :param train_portion: Portion of trainig data [0-1]
    :param val_portion: Portion of validation data [0-1]
    :param seed: RNG seed
    :returns: Tuple of train, validation, test masks
    """
    y = data.y.cpu().detach().numpy()
    unique, counts = np.unique(y, return_counts=True)
    rng = np.random.default_rng(seed)
    train = []
    val = []
    test = []
    for cl in unique:
        tmp = np.argwhere(y==cl)
        c1 = int(len(tmp)*train_portion)
        c2 = int(len(tmp)*(train_portion+val_portion))
        rng.shuffle(tmp)
        train.append(tmp[:c1])
        val.append(tmp[c1:c2])
        test.append(tmp[c2:])
    train_ix = np.concatenate(train)
    val_ix = np.concatenate(val)
    test_ix = np.concatenate(test)
    train_mask = torch.full_like(data.y, False, dtype=torch.bool)
    train_mask[train_ix] = True
    val_mask = torch.full_like(data.y, False, dtype=torch.bool)
    val_mask[val_ix] = True
    test_mask = torch.full_like(data.y, False, dtype=torch.bool)
    test_mask[test_ix] = True
    return train_mask, val_mask, test_mask

def get_dataset(dataset_name, corruption=0, seed=None, calc_pagerank=True):
    """Returns the dataset of given name

    :param dataset_name: Case sensitive name of dataset
    :returns: Processed dataset
    """
    load_function = datasets[dataset_name]
    dataset_location = ''.join(["/tmp/", dataset_name])
    dataset = load_function(root=dataset_location, name=dataset_name)[0]
    dataset.edge_index = to_undirected(dataset.edge_index.to(device))
    dataset.x = dataset.x.to(device)
    dataset.y = torch.flatten(dataset.y).to(device)
    dataset.train_mask, dataset.val_mask, dataset.test_mask = create_split(dataset,seed=seed)
    dataset.train_mask = dataset.train_mask.to(device)
    dataset.val_mask = dataset.val_mask.to(device)
    dataset.test_mask = dataset.test_mask.to(device)
    dataset.propagated_mask = torch.full_like(dataset.y, False, dtype=torch.bool).to(device)
    dataset.orig_num_classes = len(dataset.y.unique())
    dataset.num_classes = n_class_estimations[dataset_name]
    if calc_pagerank:
        network = to_networkx(dataset)
        dataset.pagerank = torch.tensor(list(pagerank(network).values())).to(device)
    dataset.ground_truth = torch.clone(dataset.y)
    if corruption > 0:
        corrupted_indices = torch.multinomial(
            (dataset.train_mask | dataset.val_mask).float(),
            int(corruption * (dataset.train_mask | dataset.val_mask).sum()))
        dataset.ground_truth[corrupted_indices] = (
            dataset.ground_truth[corrupted_indices] +
            torch.randint(low=1,high=dataset.num_classes,size=corrupted_indices.shape, device=device)) % dataset.num_classes
    return dataset

def get_homophily(dataset):
    """Calculates homophily of a dataset"""
    return homophily(dataset.edge_index, dataset.y, method='edge_insensitive')

def get_class_sizes(dataset):
    """Calculates class distributions of given dataset"""
    return torch.bincount(dataset.y)

#for name in datasets.keys():
#    dataset = get_dataset(name)
#    homo = get_homophily(dataset)
#    sizes = get_class_sizes(dataset)
#    print(name, homo)
