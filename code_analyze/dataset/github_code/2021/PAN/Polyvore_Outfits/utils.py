import json
import numpy as np
import time
import scipy.sparse as sp
from scipy.sparse import csr_matrix

def construct_feed_dict(placeholders, node_features, support, labels,attribute_labels,
                        r_indices, c_indices,
                        dropout, is_train=True,edge_c=True):
    """
    Create feed dictionary.
    """

    if not type(support[0]) == tuple:
        support = [sparse_to_tuple(sup) for sup in support]

    feed_dict = dict()
    feed_dict.update({placeholders['node_features']: node_features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})

    feed_dict.update({placeholders['labels']: labels})
    if edge_c==True:
        feed_dict.update({placeholders['attribute_labels']:attribute_labels})
    feed_dict.update({placeholders['row_indices']: r_indices})
    feed_dict.update({placeholders['col_indices']: c_indices})

    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['is_train']: is_train})

    return feed_dict

def support_dropout(sup, do, edge_drop=False):
    before = time.time()
    sup = sp.tril(sup)
    assert do > 0.0 and do < 1.0
    n_nodes = sup.shape[0]
    # nodes that I want to isolate
    isolate = np.random.choice(range(n_nodes), int(n_nodes*do), replace=False)
    nnz_rows, nnz_cols = sup.nonzero()

    # mask the nodes that have been selected
    mask = np.in1d(nnz_rows, isolate)
    mask += np.in1d(nnz_cols, isolate)
    assert mask.shape[0] == sup.data.shape[0]

    sup.data[mask] = 0
    sup.eliminate_zeros()

    if edge_drop:
        prob = np.random.uniform(0, 1, size=sup.data.shape)
        remove = prob < do
        sup.data[remove] = 0
        sup.eliminate_zeros()

    sup = sup + sup.transpose()
    return sup

def write_log(data, logfile):
    with open(logfile, 'w') as outfile:
        json.dump(data, outfile)

def get_degree_supports(adj, k, adj_self_con=False, verbose=True):
    if verbose:
        print('Computing adj matrices up to {}th degree'.format(k))
    supports = [sp.identity(adj.shape[0])]#对角线单位矩阵
    if k == 0: # return Identity matrix (no message passing)
        return supports
    assert k > 0
    supports = [sp.identity(adj.shape[0]), adj.astype(np.float64) + adj_self_con*sp.identity(adj.shape[0])]
    #supports: adjency matrix + identity matrix
    prev_power = adj
    for i in range(k-1):
        pow = prev_power.dot(adj)
        new_adj = ((pow) == 1).astype(np.float64)
        new_adj.setdiag(0)
        new_adj.eliminate_zeros()
        supports.append(new_adj)
        prev_power = pow
    return supports

def normalize_nonsym_adj(adj):
    degree = np.asarray(adj.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf

    degree_inv_sqrt = 1. / np.sqrt(degree)
    degree_inv_sqrt_mat = sp.diags([degree_inv_sqrt], [0])

    degree_inv = degree_inv_sqrt_mat.dot(degree_inv_sqrt_mat)

    adj_norm = degree_inv.dot(adj)

    return adj_norm

def sparse_to_tuple(sparse_mx):
    """ change of format for sparse matrix. This format is used
    for the feed_dict where sparse matrices need to be linked to placeholders
    representing sparse matrices. """

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def generate_attr_labels(train_c_indices,train_r_indices,n_output=20,
attribute_list = ["leather","black","high","white","denim",
"cotton","wide","blue","ankle","strappy","short","long","skinny","red","golden",
"pink","buttoned","knitted","striped"]):
    attr2idx = {}
    idx2attr = {}
    with open("data/polyvore_outfits/dataset/attribute_shuffle_list.txt","r") as f:
        idx = 0
        for ele in f:
            attr2idx[ele[:-1]] = idx
            idx2attr[idx] = ele[:-1]
            idx+=1
    
    idxs = []
    for attr in attribute_list:
        idxs.append(attr2idx[attr])
    print(idxs)
    idxs = idxs[:n_output]
    with open("data/polyvore_outfits/dataset/idx2theirattr.json",'r') as f:
        idx2theirattr = json.load(f)

   
    train_c_labels = np.zeros((train_c_indices.shape[0],n_output))
    train_r_labels = np.zeros((train_r_indices.shape[0],n_output))
    
    for i in range(len(train_c_indices)):
    #for i in range(n_output):
        c = train_c_indices[i]
        c_attr = idx2theirattr[str(int(c))]
        for j in range(len(idxs)):
            if idxs[j] in c_attr:
                train_c_labels[i,j]=1
    
    for i in range(len(train_r_indices)):
    #for i in range(n_output):
        r = train_r_indices[i]
        r_attr = idx2theirattr[str(int(r))]
        for j in range(len(idxs)):
            if idxs[j] in r_attr:
                train_r_labels[i,j]=1
    return train_c_labels,train_r_labels





class Graph(object):
    """docstring for Graph."""
    def __init__(self, adj):
        super(Graph, self).__init__()
        self.adj = adj
        self.n_nodes = adj.shape[0]
        self.level = 0

    def run_K_BFS(self, n, K):
        """
        Returns a list of K edges, sampled using BFS starting from n
        """
        visited = set()
        edges = []
        self.BFS(n, visited, K, edges)
        assert len(edges) <= K

        return edges

    def BFS(self, n, visited, K, edges):
        queue = [n]
        while len(queue) > 0:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                neighs = list(self.adj[node].nonzero()[1])
                for neigh in neighs:
                    if neigh not in visited:
                        edges.append((node, neigh))
                        queue.append(neigh)
                    if len(edges) == K:
                        return
