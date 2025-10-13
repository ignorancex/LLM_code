import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import yaml
from concurrent.futures import ThreadPoolExecutor
from HIS.globals import *


def load_data(prefix, normalize=True):
    adj_full = sp.load_npz('./{}/adj_full.npz'.format(prefix)).astype(bool)
    adj_train = sp.load_npz('./{}/adj_train.npz'.format(prefix)).astype(bool)
    # adj_train = adj_full
    adj_full = removing_multiple_and_selfloop_edges(adj_full)[0]
    role = json.load(open('./{}/role.json'.format(prefix)))
    feats = np.load('./{}/feats.npy'.format(prefix))
    class_map = json.load(open('./{}/class_map.json'.format(prefix)))

    class_map = {int(k): v for k, v in class_map.items()}
    assert len(class_map) == feats.shape[0]

    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

    num_vertices = adj_full.shape[0]

    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        labels = np.zeros((num_vertices, num_classes))
        for node, label in class_map.items():
            labels[node] = label
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        labels = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for node, label in class_map.items():
            labels[node][label - offset] = 1

    return adj_full, adj_train, feats, labels, role



def parse_n_prepare(flags, mode="train"):
    with open(flags.train_config) as f_train_config:
        train_config = yaml.safe_load(f_train_config)
    arch_gcn = {
        'dim': -1,
        'aggr': 'concat',
        'loss': 'softmax',
        'arch': '1',
        'act': 'I',
        'bias': 'norm'
    }
    arch_gcn.update(train_config['network'][0])
    train_params = {
        'lr': 0.01,
        'dropout': 0.1,
        'weight_decay': 0.,
        'norm_loss': True,
        'norm_aggr': True,
        'q_threshold': 50,
        'q_offset': 0
    }
    train_params.update(train_config['params'][0])
    train_phases = train_config['phase']
    for ph in train_phases:
        assert 'end' in ph
        assert 'sampler' in ph
    print("Loading training data..")
    if mode == "train":
        train_data = load_data(flags.data_prefix)
    else:
        raise NotImplementedError
    print("Done loading training data..")
    return train_params, train_phases, train_data, arch_gcn


def norm_aggr(data, edge_index, norm_aggr, num_proc=20):
    length = len(data)

    def process(i):
        data[i] = norm_aggr[edge_index[i]]

    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        executor.map(process, range(length))


def adj_norm(adj, deg=None, sort_indices=True, model='GCN'):
    if model=='GCN':
        diag_shape = (adj.shape[0], adj.shape[1])

        # Compute degree matrix D (or use provided degree vector)
        D = adj.sum(1).flatten() if deg is None else deg
        # Symmetric normalization: D^{-1/2}
        sqrt_deg_inv = 1.0 / np.sqrt(D)
        sqrt_deg_inv[np.isinf(sqrt_deg_inv)] = 0.0  # Handle division by zero
        # Create diagonal matrix with D^{-1/2}
        norm_diag = sp.dia_matrix((sqrt_deg_inv, 0), shape=diag_shape)
        # Perform D^{-1/2} * A * D^{-1/2}
        adj_norm = norm_diag.dot(adj).dot(norm_diag)
        # Optionally sort the indices
        if sort_indices:
            adj_norm.sort_indices()
    else:
        diag_shape = (adj.shape[0], adj.shape[1])
        D = adj.sum(1).flatten() if deg is None else deg
        norm_diag = sp.dia_matrix((1 / D, 0), shape=diag_shape)
        adj_norm = norm_diag.dot(adj)
        if sort_indices:
            adj_norm.sort_indices()

    return adj_norm



def parse_layer_yml(arch_gcn, dim_input):

    num_layers = len(arch_gcn['arch'].split('-'))
    # set default values, then update by arch_gcn
    bias_layer = [arch_gcn['bias']] * num_layers
    act_layer = [arch_gcn['act']] * num_layers
    aggr_layer = [arch_gcn['aggr']] * num_layers
    dims_layer = [arch_gcn['dim']] * num_layers
    order_layer = [int(o) for o in arch_gcn['arch'].split('-')]
    return [dim_input] + dims_layer, order_layer, act_layer, bias_layer, aggr_layer


def log_dir(f_train_config, prefix, timestamp):
    log_dir = args_global.dir_log+"/log_train/" + prefix.split("/")[-1]
    log_dir += "/{ts}-{model}/".format(model='mycode', ts=timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if f_train_config != '':
        from shutil import copyfile
        copyfile(f_train_config,'{}/{}'.format(log_dir,f_train_config.split('/')[-1]))
    return log_dir


def removing_multiple_and_selfloop_edges(graph):
    csr_adj_T = graph.transpose()
    _ = 0
    if (graph != csr_adj_T).nnz != 0:
        graph = graph.maximum(graph.T)
        graph.setdiag(0)
        graph.eliminate_zeros()
        _ = 1.2e-2
    return graph, _


_bcolors = {'header': '\033[95m',
            'blue': '\033[94m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m'}


def printf(msg,style=''):
    if not style or style == 'black':
        print(msg)
    else:
        print("{color1}{msg}{color2}".format(color1=_bcolors[style],msg=msg,color2='\033[0m'))



