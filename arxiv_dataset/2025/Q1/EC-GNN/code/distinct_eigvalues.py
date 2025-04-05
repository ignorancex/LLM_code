# from data import get_dataset
import networkx as nx
import torch
import torch.nn as nn
import datasets
from torch_geometric.data import Data
from torch_geometric.datasets import FakeDataset,FakeHeteroDataset
import numpy as np
from torch_geometric.utils import *
import matplotlib.pyplot as plt
import math
np.set_printoptions(formatter={'float': '{:0.5f}'.format})
path = '/data'
device = 'cpu'
dataset_list = [ 'cora', 'citeseer', 'pubmed', 'computers', 'photo', 'texas','cornell', 'chameleon', 'film', 'squirrel']


def distinct_num(dataset):
    eps = 1e-6
    count  = 0
    data = datasets.load_dataset(dataset, 'dense')
    data.edge_index = to_undirected(data.edge_index)
    data.edge_index, data.edge_attr = remove_self_loops(data.edge_index)
    A =  to_dense_adj(data.edge_index).squeeze()
    I = torch.eye(A.shape[0])
    degree = torch.diag(A.sum(-1)**(-0.5))
    degree[torch.isinf(degree)] = 0.
    L_sym = I - degree.mm(A.mm(degree))
    e,_=torch.symeig(L_sym,False)
    for i in range(1,len(e.numpy())):
        if torch.abs(e[i]-e[i-1])<eps:
            count = count+1
    print('dataset','node_num','distinct_num')
    print(dataset,data.x.shape[0],data.edge_index.shape[1],data.x.shape[0]-count)

distinct_num('cora')
    

