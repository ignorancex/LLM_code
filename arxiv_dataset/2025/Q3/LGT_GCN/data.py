import os.path
import torch
import numpy as np
import scipy.sparse as sp
from dgl.data import CoauthorCSDataset, AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, CoraFullDataset, \
    CoauthorPhysicsDataset, CitationGraphDataset

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str,round_no=0):
    round_index_file = f'./data/{dataset_str}_split_{round_no}.npz'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset_str in ["cora", "citeseer", "pubmed"]:
        dataset = CitationGraphDataset(dataset_str,raw_dir='./data')
    elif dataset_str in ['CoauthorCS', 'AmazonPhoto','CoraFull','AmazonComputer','CoauthorPhysics']:
        if dataset_str == 'CoauthorCS':
            dataset = CoauthorCSDataset('./data')
        elif dataset_str == 'AmazonPhoto':
            dataset = AmazonCoBuyPhotoDataset('./data')
        elif dataset_str == "CoraFull":
            dataset = CoraFullDataset('./data')
        elif dataset_str == 'AmazonComputer':
            dataset = AmazonCoBuyComputerDataset('./data')
        elif dataset_str == 'CoauthorPhysics':
            dataset = CoauthorPhysicsDataset('./data')
    data = dataset[0]
    num_class = dataset.num_classes
    feat = data.ndata['feat']
    label = data.ndata['label']
    adj = data.adjacency_matrix().to_dense()+ torch.eye(feat.shape[0])
    adj =  sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))
    if not os.path.exists(round_index_file):
        np.random.seed(round_no)
        node_range = list(range(0, int(feat.shape[0])))
        np.random.shuffle(node_range)
        test_mask = node_range[:1000]
        val_mask = node_range[1000:1000+1000]
        train_mask = get_per_class_node2(node_range[2000:], label, num_class, 20)
        np.savez(round_index_file, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)
    else:
        mask = np.load(round_index_file)
        train_mask = mask['train_mask']
        test_mask = mask['test_mask']
        val_mask = mask['val_mask']

    data.x = feat.to(device)
    data.adj = adj.to(device)
    data.edge_index = data.edges()
    data.y = label.to(device)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


def get_per_class_node2(range_node,label,num_class,num_per=20):
    arr= [0]*num_class
    mask=[]
    for i in range_node:
       if  arr[label[i]] < num_per and sum(arr)<num_per*num_class:
           mask.append(i)
           arr[label[i]] += 1
    return mask

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_adj_row(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(adj)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).to_sparse_csr()

def normalize_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()
