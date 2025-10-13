import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
import sys
from scipy.sparse import save_npz
import json

"""
This file is used to convert the FastGCN format dataset into the dataset format required by our code. 
Place this file in the ./FastGCN/data directory and execute it directly
The dataset needs to be specified in "__main__" below
"""

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)

def load_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(ally)-500)
    idx_val = range(len(ally)-500, len(ally))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels



 # 'citeseer', 'pubmed'
if __name__ == "__main__":
    data_name = 'citeseer'
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data(data_name)

    os.makedirs(data_name, exist_ok=True)

    adj_full = adj
    G = nx.from_scipy_sparse_array(adj_full)
    G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))
    adj_full = nx.to_scipy_sparse_array(G, format='csr')
    adj_full.eliminate_zeros()
    save_npz(os.path.join(data_name, "adj_full.npz"), adj_full)

    train_index = np.where(train_mask)[0]
    train_mask = train_mask.astype(bool)

    adj_train = adj.copy()
    adj_train[~train_mask, :] = 0
    adj_train[:, ~train_mask] = 0
    adj_train.eliminate_zeros()
    save_npz(os.path.join(data_name, "adj_train.npz"), adj_train)

    role = {
        'tr': train_index.tolist(),
        'va': np.where(val_mask)[0].tolist(),
        'te': np.where(test_mask)[0].tolist()
    }
    with open(os.path.join(data_name, "role.json"), "w") as f:
        json.dump(role, f)

    y = np.array(labels)
    if y.shape[1] > 1:
        class_map = {i: y[i].tolist() for i in range(len(y))}
    else:
        class_map = {i: int(y[i]) for i in range(len(y))}
    with open(os.path.join(data_name, "class_map.json"), "w") as f:
        json.dump(class_map, f)


    features_dense = features.todense()
    np.save(os.path.join(data_name, "feats.npy"), features_dense)
    print("success！")











