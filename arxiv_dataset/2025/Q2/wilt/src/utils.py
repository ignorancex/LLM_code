import os

import numpy as np
import torch
from torch_geometric.data import Data, Dataset  # type: ignore
from torch_geometric.datasets import ZINC, MoleculeNet, TUDataset  # type: ignore
from torch_geometric.transforms import Constant

from path import DATA_DIR


def load_dataset(dataset_name: str) -> Dataset:
    if dataset_name in ["synthetic_bin", "synthetic_mul", "synthetic_reg"]:
        data = torch.load(
            os.path.join(DATA_DIR, "synthetic", dataset_name[-3:], "dataset.pt")
        )
    elif dataset_name == "ZINC":
        data = ZINC(root=os.path.join(DATA_DIR, "ZINC"), subset=True, split="train")
    elif dataset_name in ["Lipo", "ESOL"]:
        data = MoleculeNet(root=os.path.join(DATA_DIR, dataset_name), name=dataset_name)
        converted = []
        xdict = {}
        edict = {}
        for d in data:
            newx = torch.zeros(len(d.x), 1, dtype=torch.int32)
            for i, x in enumerate(d.x):
                idx = tuple(x.tolist())
                if idx not in xdict:
                    xdict[idx] = len(xdict)
                newx[i][0] = xdict[idx]
            newe = torch.zeros(len(d.edge_attr), 1, dtype=torch.int32)
            for i, e in enumerate(d.edge_attr):
                idx = tuple(e.tolist())
                if idx not in edict:
                    edict[idx] = len(edict)
                newe[i][0] = edict[idx]
            converted.append(
                Data(x=newx, edge_attr=newe, edge_index=d.edge_index, y=d.y)
            )
        data = converted
    elif dataset_name in ["IMDB-BINARY", "COLLAB"]:
        data = TUDataset(
            root=os.path.join(DATA_DIR, "TUDataset"),
            name=dataset_name,
            pre_transform=Constant(),
        )
    else:
        data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    return data


def calc_rmse_wo_outliers(x1: np.ndarray, y1: np.ndarray) -> tuple:
    x1 /= np.max(x1)
    y1 /= np.max(y1)
    c1 = np.sum(x1 * y1) / np.sum(x1**2)
    diff = np.abs(y1 - c1 * x1)
    sorted_indices = np.argsort(diff)
    n_outliers = len(sorted_indices) // 100
    x = x1[sorted_indices[:-n_outliers]]
    y = y1[sorted_indices[:-n_outliers]]
    x /= np.max(x)
    y /= np.max(y)
    coeff = np.sum(x * y) / np.sum(x**2)
    return x, y, coeff, np.sqrt(np.mean((y - coeff * x) ** 2))
