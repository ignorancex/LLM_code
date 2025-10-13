import os
import sys
import numpy as np
import random
import torch
import torch.optim as optim
import networkx as nx
from tqdm import tqdm
import sys

## Functions on graph nodes
def compute_num_nodes(batched_graphs):
    '''
    Computes the total number of nodes in a set of batched (lower-half) adjacency vectors.
    '''
    if torch.sum(batched_graphs < 0) == 0:
        ### All graphs are the same size
        N = batched_graphs.shape[0]
        num_edges = batched_graphs.shape[1]
        num_nodes = int(0.5 + (2 * num_edges + 0.25)**0.5)
        return N * num_nodes
    else:
        num_nodes = 0
        for idx in range(batched_graphs.shape[0]):
            G = batched_graphs[idx]
            num_edges = torch.sum(G > -1).item()
            num_nodes += int(0.5 + (2 * num_edges + 0.25)**0.5)
        return num_nodes

def get_node_dist(graphs):
    '''
    Obtains representation of node distribution over training graphs. From BiGG.
    '''
    num_node_dist = np.bincount([len(gg.nodes) for gg in graphs])
    num_node_dist = num_node_dist / np.sum(num_node_dist)
    return num_node_dist


## Functions for converting graphs to tensors & vice-versa
def adj_vec(g, weighted, max_num_nodes):
    '''
    Transforms nx graph into adjacency vector
    Args:
        weighted: if True, returns weighted adjacency vector.
        max_num_nodes: Maximum number of nodes in graph
    '''
    n = len(g)
    A = nx.to_numpy_array(g, nodelist=sorted(g), weight='weight' if weighted else None)
    
    if n < max_num_nodes:
        pad = max_num_nodes - n
        A = np.pad(A, ((0, pad), (0, pad)), mode = 'constant', constant_values = -1)
    
    tril_indices = np.tril_indices(max_num_nodes, k=-1)
    U = A[tril_indices]

    return torch.tensor(U, dtype=torch.float32)

def graph_from_adj(vec):
    '''
    Transforms (weighted) adjacency vector into (weighted) graph
    Args:
        vec: adjacency vector to be converted. Can be weighted or unweighted.
    '''
    vec = np.array(vec)
    
    adj_len = len(vec)
    n = int((2*adj_len + .25)**0.5 + 0.5)
    tri = np.zeros((n, n))
    tri[np.tril_indices(n, k = -1)] = vec
    tri = np.transpose(tri)
    tri[np.tril_indices(n, k = -1)] = vec
    
    g = nx.from_numpy_array(tri)
    return g

# ## FIX ALL THE BATCH TRAINING STUFF...
# 
# ## create graphs ready to be batched..
def batch_preparation(train_graphs, weighted):
    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    adj_list = []
    
    for gg in train_graphs:
        U = adj_vec(gg, weighted, max_num_nodes)
        if len(U.shape) == 1:
            U = U.unsqueeze(0)
        adj_list.append(U)
    
    batched_train_graphs = torch.stack(adj_list, dim = 0)
    return batched_train_graphs.unsqueeze(-1)


    
    
    
    
    
    
    
    
    
    
    
    
    