import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pandas as pd
import numpy as np

def load_network_data(adj_name):
    # if adj_name == 'cora':
    #     nodes_numbers = 2708
    #     raw_edges = pd.read_csv("datasets/cora-edges.txt",header=None)
    # elif adj_name == 'wiki':
    #     nodes_numbers = 2405
    #     raw_edges = pd.read_csv('datasets/graph.txt', header=None, sep='\t')
    # elif adj_name == 'email':
    #     nodes_numbers = 1133
    #     raw_edges = pd.read_csv('datasets/ia-email-univ.mtx', header=None, sep=' ') - 1
    # elif adj_name == 'citeseer':
    #     nodes_numbers = 3327
    #     raw_edges = pd.read_csv('datasets/citeseer-edges.txt', header=None)
    # else:
    #     print("Dataset is not exist!")

    if adj_name == 'cora':
        nodes_number = 2708    
    elif adj_name == 'citeseer':
        nodes_number = 3327    
    elif adj_name == 'wiki':
        nodes_number = 2405    
    elif adj_name == 'celegans':
        nodes_number = 297    
    elif adj_name == 'email':
        nodes_number = 986    
    elif adj_name == 'polbooks':
        nodes_number = 105    
    elif adj_name == 'texas':
        nodes_number = 183    
    elif adj_name == 'wisconsin':
        nodes_number = 251
    elif adj_name == 'karate':
        nodes_number = 34
        adj_name = 'karate_edges'

    raw_edges = pd.read_csv("data/"+adj_name+".txt",header=None,sep=' ')

    drop_self_loop = raw_edges[raw_edges[0] != raw_edges[1]]

    graph_np = np.zeros((nodes_number, nodes_number))

    for i in range(drop_self_loop.shape[0]):
        graph_np[drop_self_loop.iloc[i,0], drop_self_loop.iloc[i,1]]=1
        graph_np[drop_self_loop.iloc[i,1], drop_self_loop.iloc[i,0]]=1

    adj = nx.adjacency_matrix(nx.from_numpy_matrix(graph_np))

    features = sp.lil_matrix(np.eye(nodes_number))
    
    return adj, features


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_test, tx, ty, test_mask, np.argmax(labels,1)


def load_alldata(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset_str, names[i]))))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
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
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, np.argmax(labels, 1)
