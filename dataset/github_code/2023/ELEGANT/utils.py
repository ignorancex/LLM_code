import numpy as np
import scipy.sparse as sp
import torch
import sys

import pickle as pkl
import networkx as nx
import os
import dgl
import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance_matrix
import csv

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot



def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map = np.array(idx_map)

    return idx_map

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def init_logfile(filename, text):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()

def log(filename, text):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data_new(dataset, alpha=0.0, n_iter=4): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    """
    Loads input data from gcn/data directory

    ind.dataset.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset.ally => the labels for instances in ind.dataset.allx as numpy.ndarray object;
    ind.dataset.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    #print(graph)
    print(allx.shape,tx.shape)
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
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    #print(features.shape)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #print(labels.shape)

    idx_test = test_idx_range.tolist()
    
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    #print('#test:', len(idx_test), )


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if alpha == 0.0:
        features = normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
    else:
        features = normalize_prop(features, adj, alpha, n_iter, normFea=False)
        features = torch.FloatTensor(np.array(features))

    #labels = torch.LongTensor(np.where(labels)[1])
    labels = torch.LongTensor(np.argmax(labels,1))

    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)


    return adj, features, labels, idx_train, idx_val, idx_test


def load_data(path="../data/cora/", dataset="../data/cora", alpha=0.0, n_iter=4):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    if dataset == "cora":

        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if alpha == 0.0:
        features = normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
    else:
        features = normalize_prop(features, adj, alpha, n_iter, normFea=False)
        features = torch.FloatTensor(np.array(features))

    adj = normalize(adj + sp.eye(adj.shape[0]))

    if dataset == 'cora':
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

    if dataset == 'citeseer':
        pass

    if dataset == 'pubmed':
        pass

    #features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    #r_mat_inv = sp.diags(r_inv)
    r_mat_inv = sp.diags(r_inv,0)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_prop(mx, adj, alpha, n_iter, normFea=False):

    if normFea:
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        # print(r_inv.shape)
        # print(r_inv)
        r_mat_inv = sp.diags(r_inv,0)
        #r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
    else:
        mx = mx.todense()

    """Feature propagation via Normalized Laplacian"""
    S = normalize_adj(adj)
    F = alpha * S.dot(mx) + (1-alpha) * mx
    for _ in range(n_iter):
        F = alpha * S.dot(F) + (1-alpha) * mx
    return F


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)





def load_income(dataset, sens_attr="race", predict_attr="income", path="data/income/", label_number=1e10):  # 1000
    print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    # adj = sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    # features = torch.rand(features.shape[0], features.shape[1])
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]

    features[label_idx_0, :] = (features[label_idx_0, :] - features[label_idx_0, :].mean(axis=0)) / features[label_idx_0, :].std(axis=0)
    features[label_idx_1, :] = (features[label_idx_1, :] - features[label_idx_1, :].mean(axis=0)) / features[label_idx_1, :].std(axis=0)

    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)


    smaller_num = min(len(label_idx_0), len(label_idx_1))
    idx_train = np.append(label_idx_0[:int(0.5 * smaller_num)], label_idx_1[:int(0.5 * smaller_num)])
    idx_val = np.append(label_idx_0[int(0.5 * smaller_num):int(0.8 * smaller_num)], label_idx_1[int(0.5 * smaller_num):int(0.8 * smaller_num)])
    idx_test = np.append(label_idx_0[int(0.8 * smaller_num):int(1 * smaller_num)], label_idx_1[int(0.8 * smaller_num):int(1 * smaller_num)])


    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)



    return adj, features, labels, idx_train, idx_val, idx_test, sens



def load_pokec_renewed(dataset, label_number=1000):  # 1000

    if dataset == 1:
        edges = np.load('data/pokec_dataset/region_job_1_edges.npy')
        features = np.load('data/pokec_dataset/region_job_1_features.npy')
        labels = np.load('data/pokec_dataset/region_job_1_labels.npy')
        sens = np.load('data/pokec_dataset/region_job_1_sens.npy')
    else:
        edges = np.load('data/pokec_dataset/region_job_2_2_edges.npy')
        features = np.load('data/pokec_dataset/region_job_2_2_features.npy')
        labels = np.load('data/pokec_dataset/region_job_2_2_labels.npy')
        sens = np.load('data/pokec_dataset/region_job_2_2_sens.npy')

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]

    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens




def load_facebook():
    edges_file=open('data/facebook/facebook/107.edges')
    edges=[]
    for line in edges_file:
        edges.append([int(one) for one in line.strip('\n').split(' ')])

    feat_file=open('data/facebook/facebook/107.feat')
    feats=[]
    for line in feat_file:
        feats.append([int(one) for one in line.strip('\n').split(' ')])

    feat_name_file = open('data/facebook/facebook/107.featnames')
    feat_name = []
    for line in feat_name_file:
        feat_name.append(line.strip('\n').split(' '))
    names={}
    for name in feat_name:
        if name[1] not in names:
            names[name[1]]=name[1]
        if 'gender' in name[1]:
            print(name)

    print(feat_name)
    feats=np.array(feats)

    node_mapping={}
    for j in range(feats.shape[0]):
        node_mapping[feats[j][0]]=j

    feats=feats[:,1:]

    print(feats.shape)
    #for i in range(len(feat_name)):
    #    print(i, feat_name[i], feats[:,i].sum())

    sens=feats[:,264]


    labels=feats[:,220]
    feats=np.concatenate([feats[:,:264],feats[:,266:]],-1)
    feats=np.concatenate([feats[:,:220],feats[:,221:]],-1)


    # labels=feats[:,list(range(75, 89))].argmax(1)
    # print(labels)
    # feats=np.concatenate([feats[:,:264],feats[:,266:]],-1)
    # feats=np.concatenate([feats[:,:75],feats[:,89:]],-1)



    edges=np.array(edges)
    #edges=torch.tensor(edges)
    #edges=torch.stack([torch.tensor(one) for one in edges],0)
    print(len(edges))

    node_num=feats.shape[0]
    adj=np.zeros([node_num,node_num])


    for j in range(edges.shape[0]):
        adj[node_mapping[edges[j][0]],node_mapping[edges[j][1]]]=1


    idx_train=np.random.choice(list(range(node_num)),int(0.8*node_num),replace=False)
    idx_val=list(set(list(range(node_num)))-set(idx_train))
    idx_test=np.random.choice(idx_val,len(idx_val)//2,replace=False)
    idx_val=list(set(idx_val)-set(idx_test))

    features = torch.FloatTensor(feats)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    labels = torch.LongTensor(labels)

    features=torch.cat([features,sens.unsqueeze(-1)],-1)
    adj=sp.csr_matrix(adj)
    return adj, features, labels, idx_train, idx_val, idx_test, sens



def load_LCC(dataset_name='LCC_small'):
    if dataset_name=='LCC':
        path='./data/raw_LCC'
        name='LCC'
    elif dataset_name=='LCC_small':
        path='./data/raw_Small'
        name='Small'
    else:
        raise NotImplementedError

    edgelist=csv.reader(open(path+'/edgelist_{}.txt'.format(name)))

    edges=[]
    for line in edgelist:
        edge=line[0].split('\t')
        edges.append([int(one) for one in edge])


    edges=np.array(edges)
    print(len(edges))

    labels_file=csv.reader(open(path+'/labels_{}.txt'.format(name)))
    labels=[]
    for line in labels_file:
        labels.append(float(line[0].split('\t')[1]))
    labels=np.array(labels)

    sens_file=csv.reader(open(path+'/sens_{}.txt'.format(name)))
    sens=[]
    for line in sens_file:
        sens.append([float(line[0].split('\t')[1])])
    sens=np.array(sens)


    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    node_num=labels.shape[0]
    #adj=adj.todense()
    idx_train=np.random.choice(list(range(node_num)),int(0.8*node_num),replace=False)
    idx_val=list(set(list(range(node_num)))-set(idx_train))
    # idx_test=np.random.choice(idx_val,len(idx_val)//2,replace=False)
    idx_test=np.random.choice(idx_val,len(idx_val)//10,replace=False)
    idx_val=list(set(idx_val)-set(idx_test))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    labels = torch.LongTensor(labels)
    sens = torch.FloatTensor(sens)
    features=np.load(path+'/X_{}.npz'.format(name))


    features=torch.FloatTensor(sp.coo_matrix((features['data'], (features['row'], features['col'])),
                  shape=(labels.shape[0], np.max(features['col'])+1),
                  dtype=np.float32).todense())
    features = torch.cat([features, sens], -1)
    return adj, features, labels, idx_train, idx_val, idx_test, sens



def load_bail(dataset='bail', sens_attr="WHITE", predict_attr="RECID", path="data/bail/", label_number=1000):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])


    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    # random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]

    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)


    idx_train = np.append(label_idx_0[:min(int(0.4 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.4 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.4 * len(label_idx_0)):int(0.95 * len(label_idx_0))],
                        label_idx_1[int(0.4 * len(label_idx_1)):int(0.95 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.95 * len(label_idx_0)):], label_idx_1[int(0.95 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens




def load_credit(dataset='credit', sens_attr="Age", predict_attr="NoDefaultNextMonth", path="data/credit/",
                label_number=6000):
    from scipy.spatial import distance_matrix

    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')


    # build relationship
    edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')


    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)

    print(len(edges))

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.4 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.4 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.4 * len(label_idx_0)):int(0.95 * len(label_idx_0))],
                        label_idx_1[int(0.4 * len(label_idx_1)):int(0.95 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.95 * len(label_idx_0)):], label_idx_1[int(0.95 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens

def load_german(dataset="german", sens_attr="Gender", predict_attr="GoodCustomer", path="data/german/",
                label_number=100):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

    edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)

    print(len(edges))

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])



    # num0 = int(0.75 * len(label_idx_0))
    # num1 = int(0.75 * len(label_idx_1))
    # flag_num = min(num0, num1)
    # idx_test = np.append(label_idx_0[-flag_num:], label_idx_1[-flag_num:])


    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens





def load_google():
    id='111058843129764709244'
    edges_file=open('data/gplus/{}.edges'.format(id))
    edges=[]
    for line in edges_file:
        edges.append([int(one) for one in line.strip('\n').split(' ')])

    feat_file=open('data/gplus/{}.feat'.format(id))
    feats=[]
    for line in feat_file:
        feats.append([int(one) for one in line.strip('\n').split(' ')])

    feat_name_file = open('data/gplus/{}.featnames'.format(id))
    feat_name = []
    for line in feat_name_file:
        feat_name.append(line.strip('\n').split(' '))
    names={}
    for name in feat_name:
        if name[1] not in names:
            names[name[1]]=name[1]
        if 'gender' in name[1]:
            print(name)

    #print(feat_name)
    feats=np.array(feats)

    node_mapping={}
    for j in range(feats.shape[0]):
        node_mapping[feats[j][0]]=j

    feats=feats[:,1:]

    print(feats.shape)
    for i in range(len(feat_name)):
        if feats[:,i].sum()>100:
            print(i, feat_name[i], feats[:,i].sum())

    feats=np.array(feats,dtype=float)


    sens=feats[:,0]
    labels=feats[:,164]


    feats=np.concatenate([feats[:,:164],feats[:,165:]],-1)
    feats=feats[:,1:]



    edges=np.array(edges)
    #edges=torch.tensor(edges)
    #edges=torch.stack([torch.tensor(one) for one in edges],0)

    print(len(edges))

    node_num=feats.shape[0]
    adj=np.zeros([node_num,node_num])


    for j in range(edges.shape[0]):
        adj[node_mapping[edges[j][0]],node_mapping[edges[j][1]]]=1


    idx_train=np.random.choice(list(range(node_num)),int(0.8*node_num),replace=False)
    idx_val=list(set(list(range(node_num)))-set(idx_train))
    idx_test=np.random.choice(idx_val,len(idx_val)//2,replace=False)
    idx_val=list(set(idx_val)-set(idx_test))

    features = torch.FloatTensor(feats)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    labels = torch.LongTensor(labels)
    features = torch.cat([features, sens.unsqueeze(-1)], -1)
    adj1 = sp.csr_matrix(adj)

    return adj1, features, labels, idx_train, idx_val, idx_test, sens



















