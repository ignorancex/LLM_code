import numpy as np
import scipy.sparse as sps
import torch
import random

def relabel_node_ids(df, relabel=True):
    unique_src_ids = df['source_node_id'].unique()
    unique_dst_ids = df['partner_node_id'].unique()
    unique_ids = np.unique(np.concatenate((unique_src_ids, unique_dst_ids), axis=None))
    empty_partner = -1 in unique_ids
    if relabel:
        node_id_set = {node_id: i for i, node_id in enumerate(unique_ids)}

        assign_node_id = lambda x: node_id_set[x]

        df['source_node_id'] = df['source_node_id'].apply(assign_node_id)
        df['partner_node_id'] = df['partner_node_id'].apply(assign_node_id)
    n_nodes = len(unique_ids)

    return df, n_nodes - 1 if empty_partner else n_nodes


def generate_ns(positive_edges, edges_no_weights, num_nodes, ns=1):
    ns_dict = dict()
    while len(ns_dict) < len(positive_edges) * ns:
        src = random.randint(0, num_nodes-1)
        dst = random.randint(0, num_nodes-1)
        if src!=dst \
                and (src,dst) not in ns_dict \
                and (src,dst) not in edges_no_weights:
                ns_dict[(src,dst)] = 0
    return [(src,dst,0) for (src,dst),value in ns_dict.items()]

def normalize_adj(adj):
    train_adj = sps.csr_matrix(adj, dtype=float)
    rowsum = np.array(train_adj.sum(1))
    degree_mat_inv_sqrt = sps.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = train_adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo().astype(
            np.float32)

    return adj_normalized


def generate_adjacency_matrix(edges, num_nodes):
    adj = np.zeros((num_nodes, num_nodes))
    for edge in edges:
        src = edge[0]
        dst = edge[1]
        weight = edge[2]
        adj[src][dst] = weight
    for i in range(num_nodes):
        adj[i][i] = 1
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx, device):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device=device)