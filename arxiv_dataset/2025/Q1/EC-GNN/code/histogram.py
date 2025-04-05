# from data import get_dataset
import networkx as nx
import torch
import datasets
from torch_geometric.data import Data
from torch_geometric.datasets import FakeDataset,FakeHeteroDataset
from torch.nn.init import orthogonal_
import numpy as np
from torch_geometric.utils import *
import matplotlib.pyplot as plt
import math
np.set_printoptions(formatter={'float': '{:0.5f}'.format})
path = '/data'
device = 'cpu'
dataset_list = ['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'texas', 'cornell', 'chameleon', 'film', 'squirrel' ]


def draw_density(dataset):
    data = datasets.load_dataset(dataset, 'dense')
    edge_index = to_undirected(data.edge_index)
    A =  to_dense_adj(edge_index).squeeze()
    n = A.shape[0]
    data.edge_index, data.edge_attr = remove_self_loops(data.edge_index)
    print(contains_self_loops(data.edge_index))
    I = torch.eye(A.shape[0]).to(device) 
    degree = torch.diag(A.sum(-1)**(-0.5))
    degree[torch.isinf(degree)] = 0.
    L_sym = I - degree.mm(A.mm(degree))
    e,_=torch.symeig(L_sym,eigenvectors=True)
    dri = torch.histc(e, 100, -0.001, 2.001) / L_sym.shape[0]
    x = np.arange(0, 2, 0.02)
    plt.plot(x, dri.cpu().numpy())
    plt.tick_params(labelsize=17)
    plt.xlabel('λ',fontsize=18)
    plt.ylabel('desity',fontsize=18)
    plt.savefig(dataset+'.pdf', bbox_inches='tight')
    print(e.numpy())

draw_density('cora')