import torch
from torch import Tensor
import torch.nn as nn
from mlp_soft import MLP_Parent_XenterLoss as MLP_layer
from dgl.udf import EdgeBatch, NodeBatch
from dgl.heterograph import DGLGraph


class ConvolutionGGN_layer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
        self.act_fn = nn.ReLU()

    def message_func(self, edges: EdgeBatch):
        Bx_j = edges.src['BX']
        e_j = edges.data['CE'] + edges.src['DX'] + edges.dst['EX']
        edges.data['E'] = e_j
        return {'Bx_j': Bx_j, 'e_j': e_j}

    def reduce_func(self, nodes: NodeBatch):
        Ax = nodes.data['AX']
        Bx_j = nodes.mailbox['Bx_j']
        e_j = nodes.mailbox['e_j']
        h = Ax + torch.sum(e_j * Bx_j, dim=1)
        h = self.act_fn(h)

        return {'H': h}

    def forward(self, g: DGLGraph, X: Tensor, E_X: Tensor):

        g.ndata['H'] = X
        g.ndata['AX'] = self.A(X)
        g.ndata['BX'] = self.B(X)
        g.ndata['DX'] = self.D(X)
        g.ndata['EX'] = self.E(X)
        g.edata['E'] = E_X
        g.edata['CE'] = self.C(E_X)

        g.update_all(self.message_func, self.reduce_func)

        H = g.ndata['H']  # result of graph convolution
        E = g.edata['E']  # result of graph convolution

        H = self.bn_node_h(H)  # batch normalization
        E = self.bn_node_e(E)  # batch normalization

        H = X + H  # residual connection
        E = E_X + E  # residual connection

        H = torch.relu(H)  # non-linear activation
        E = torch.relu(E)  # non-linear activation

        return H, E


class ConvolutionGGN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_feature_dim = input_dim
        self.output_dim = output_dim

        h_dim_list = []
        layer_size = input_dim
        while (layer_size >= 32):
            h_dim_list.append(layer_size)
            layer_size //= 4

        e_dim_list = h_dim_list.copy()
        h_dim_list.insert(0, self.input_feature_dim)
        e_dim_list.insert(0, 1)

        self.h_dim_list = h_dim_list
        self.e_dim_list = e_dim_list

        self.embedding_h_layers = nn.ModuleList([
            nn.Linear(self.h_dim_list[idx], self.h_dim_list[idx+1]) for idx in range(len(self.h_dim_list)-1)
        ])
        self.embedding_e_layers = nn.ModuleList([
            nn.Linear(self.e_dim_list[idx], self.e_dim_list[idx+1]) for idx in range(len(self.e_dim_list)-1)
        ])
        self.ConvolutionGGN_layers = nn.ModuleList([
            ConvolutionGGN_layer(self.h_dim_list[idx+1], self.h_dim_list[idx+1]) for idx in range(len(self.h_dim_list)-1)
        ])

        self.mlp_input_dim_parent = sum(self.h_dim_list[1:])
        self.mlp_input_dim_child = sum(self.h_dim_list[1:])

        self.MLP_layer = MLP_layer(
            self.mlp_input_dim_parent, self.output_dim)

    def forward(self, g:DGLGraph, X:Tensor, E:Tensor):
        H:Tensor =X  
        parent_gnn_feature_lst = []
        child_gnn_feature_lst = []

        for i in range(len(self.h_dim_list)-1):

            H = self.embedding_h_layers[i](H)
            E = self.embedding_e_layers[i](E)
            H, E = self.ConvolutionGGN_layers[i](g, H, E)

            num_features = H.shape[1]

            parent = H.reshape(g.batch_size, -1, num_features)[:, 0:-1:2, :]
            parent = torch.mean(parent, dim=1)

            child = H.reshape(g.batch_size, -1, num_features)[:, 1:-1:2, :]
            child = torch.mean(child, dim=1)

            parent_gnn_feature_lst.append(parent)
            child_gnn_feature_lst.append(child)

        parent_mean = torch.cat(
            (parent_gnn_feature_lst[0], parent_gnn_feature_lst[1], parent_gnn_feature_lst[2], parent_gnn_feature_lst[3]), dim=1)
        child_mean = torch.cat(
            (child_gnn_feature_lst[0], child_gnn_feature_lst[1], child_gnn_feature_lst[2], child_gnn_feature_lst[3]), dim=1)

        y, center_feature = self.MLP_layer(parent_mean, child_mean)

        return y, parent_mean, child_mean, center_feature
