import tensorflow as tf
import numpy as np
import os
import pickle

from client import Client
from utils_misc import read_data

PARTITION_STRATEGY = 'iid'

def partition_with_dir(num_worker, y_train, num_classes, alpha):
    min_size = 0
    K = num_classes
    N = y_train.shape[0]

    while min_size < 10:
        idx_batch = [[] for _ in range(num_worker)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_worker))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/num_worker) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch

def read_clients_pickle(users, train_file, test_file, model):
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)

    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)
    
    clients = [Client(u, '', train_data[u], test_data[u], model) for u in users]
    return clients, train_data, test_data
    

def create_clients_by_json_data(tn_data_dir, te_data_dir, model=None):  
    users, groups, train_data, test_data = read_data(tn_data_dir, te_data_dir)
    if len(groups) == 0:
        groups = [[] for _ in users]    
    clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    return clients, train_data, test_data, users

def setup_clients_femnist(model=None, use_val_set=False):
    dataset = 'femnist'
    eval_set = 'test' if not use_val_set else 'val'  
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', eval_set)
    clients, train_data, test_data, _ = create_clients_by_json_data(train_data_dir, test_data_dir, model)
    return clients, train_data, test_data

def setup_clients_celeba(model=None, use_val_set=False):
    dataset = 'celeba'
    eval_set = 'test' if not use_val_set else 'val'  
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', eval_set)
    
    train_file = os.path.join('data', dataset, 'data', 'train',"train_data.pb")
    test_file = os.path.join('data', dataset,  'data',eval_set,  '{}_data.pb'.format(eval_set))
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        _, _, _, users =  create_clients_by_json_data(train_data_dir, 
                                                 test_data_dir, model)         
        return read_clients_pickle(users, train_file, test_file, model)
    else:
        clients, train_data, test_data, _ = create_clients_by_json_data(train_data_dir, 
                                                      test_data_dir, model)
        return clients, train_data, test_data

def setup_clients_mnist(num_workers, model=None, use_val_set=False, alpha=0.5):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    n_train = len(y_train)
    n_test = len(y_test)
    if PARTITION_STRATEGY != 'non-iid':
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, num_workers)         
    else:
        batch_idxs = partition_with_dir(num_workers, y_train, 10, alpha)    
    
    idxs_eval = np.random.permutation(n_test)
    batch_idxseval = np.array_split(idxs_eval, num_workers)
    clients = []
    # build a dict to compatiable with femnist and celeba
    for i in range(num_workers):
        train_data = {"x": 
                      x_train[np.array(batch_idxs[i])],
                     "y":
                      y_train[np.array(batch_idxs[i])]}
        test_data = {"x":
                    x_test[np.array(batch_idxseval[i])],
                    "y":
                    y_test[np.array(batch_idxseval[i])]}
        c = Client(format(i, "03"), '', train_data, test_data, model)
        clients.append(c)
        
    return clients, (x_train, y_train), (x_test, y_test)

def setup_clients_cifar10(num_workers, model=None, use_val_set=False, alpha=0.2):
    cifar = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    x_train = x_train.reshape((-1, 32, 32, 3)).astype(np.float32) / 255.0
    x_test = x_test.reshape((-1, 32, 32, 3)).astype(np.float32) / 255.0
    
    n_train = len(y_train)
    n_test = len(y_test)
#     idxs = np.random.permutation(n_train)
#     batch_idxs = np.array_split(idxs, num_workers)    
    idxs_eval = np.random.permutation(n_test)
    batch_idxseval = np.array_split(idxs_eval, num_workers)
    if PARTITION_STRATEGY != 'non-iid':
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, num_workers)         
    else:
        batch_idxs = partition_with_dir(num_workers, y_train, 10, alpha)
    clients = []
    # build a dict to compatiable with femnist and celeba    
    for i in range(num_workers):
        train_data = {"x": 
                      x_train[np.array(batch_idxs[i])],
                     "y":
                      y_train[np.array(batch_idxs[i])]}
        test_data = {"x":
                    x_test[np.array(batch_idxseval[i])],
                    "y":
                    y_test[np.array(batch_idxseval[i])]}
        c = Client(format(i, "03"), '', train_data, test_data, model)
        clients.append(c)
        
    return clients, (x_train, y_train), (x_test, y_test)    