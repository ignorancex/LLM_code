# this script is used to load data from data folder
# it can load pytorch geometric data object with edgelist list object
# also it loads the classes y
# train_mask, test_mask, validation_mask (val_mask)
# with the shortest_distance matrix.
# this script use networkx and pytorch geometric library

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import networkx as nx

def load_data(folder_name):
    edgelist = np.load(folder_name+'/edgelist.npy', allow_pickle=True)
    y = np.load(folder_name+'/y.npy', allow_pickle=True)
    train_mask = np.load(folder_name+'/train_mask.npy', allow_pickle=True).astype(bool)
    test_mask = np.load(folder_name+'/test_mask.npy', allow_pickle=True).astype(bool)
    val_mask = np.load(folder_name+'/val_mask.npy', allow_pickle=True).astype(bool)
    
    edge_index = torch.tensor(edgelist).long().t().contiguous()
    y_t = torch.tensor(y).long()
    train_mask_t = torch.tensor(train_mask)
    val_mask_t = torch.tensor(val_mask)
    test_mask_t = torch.tensor(test_mask)
    
    data = Data(edge_index=edge_index)
    data.num_nodes = edge_index.max().item() + 1
    data.y = y_t
    data.train_mask = train_mask_t
    data.val_mask = val_mask_t
    data.test_mask = test_mask_t
    
    print('Data Loaded!')
    
    return data, edgelist, y, train_mask, test_mask, val_mask

def load_data_with_x(folder_name):
    edgelist = np.load(folder_name+'/edgelist.npy', allow_pickle=True)
    y = np.load(folder_name+'/y.npy', allow_pickle=True)
    x = np.load(folder_name+'/x.npy', allow_pickle=True)
    train_mask = np.load(folder_name+'/train_mask.npy', allow_pickle=True).astype(bool)
    test_mask = np.load(folder_name+'/test_mask.npy', allow_pickle=True).astype(bool)
    val_mask = np.load(folder_name+'/val_mask.npy', allow_pickle=True).astype(bool)
    
    edge_index = torch.tensor(edgelist).long().t().contiguous()
    x_t = torch.tensor(x).float()
    y_t = torch.tensor(y).long()
    train_mask_t = torch.tensor(train_mask)
    val_mask_t = torch.tensor(val_mask)
    test_mask_t = torch.tensor(test_mask)
    
    data = Data(edge_index=edge_index)
    data.num_nodes = edge_index.max().item() + 1
    data.x = x_t
    data.y = y_t
    data.train_mask = train_mask_t
    data.val_mask = val_mask_t
    data.test_mask = test_mask_t
    
    print('Data Loaded!')
    
    return data, edgelist, x, y, train_mask, test_mask, val_mask

def load_shrtst_dist_matrix(edgelist, is_available=False, folder_name=None):
    
    G = nx.Graph()
    G.add_edges_from(edgelist)
    
    if is_available==True:
        if folder_name is None:
            print('Wrong Folder!')
            return _ , G
        shrtst_dist = np.load(folder_name+'/shrtst_dist.npy', allow_pickle=True)
    else:
        #shortest distance
        p = nx.shortest_path_length(G)
        shrtst_dist = np.zeros((len(G.nodes), len(G.nodes)))
        for p_ in p:
            source = p_[0]
            dest_dict = p_[1]
            for dest in dest_dict.keys():
                shrtst_dist[source, dest] = dest_dict[dest]
    
    return shrtst_dist, G