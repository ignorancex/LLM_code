################# SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS  ICLR 2017
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class LGT_GCN(nn.Module):
    def __init__(self, args, nfeat, nclass ,data):
        super(LGT_GCN, self).__init__()

        self.args = args
        self.hid = args.hid
        self.nclass = nclass
        self.nlayer = args.nlayer
        self.dropout = args.dropout
        self.rank = args.rank

        self.gc1 = GraphConvolution_GCN(nfeat, self.hid, self.rank) # 1433 -> 256
        self.layers = nn.ModuleList()
        for _ in range(self.nlayer - 2):
            self.layers.append(GraphConvolution_GCN(self.hid, self.hid, self.rank)) # 256 -> 256
        self.last_layer=GraphConvolution_GCN(self.hid, nclass, self.rank) # 256 -> 7

    def forward(self, x, adj, depth=-1):
        if depth > 0:
            self.gc1.lora = True
        x = F.relu(self.gc1(x, adj))  # 第一层卷积操作(2708,1433),(2708,2708)
        x = F.dropout(x, self.dropout, training=self.training)
        print(f'layer:0')
        for i, layer in enumerate(self.layers):
            i = i+1
            if depth >-1 and i < depth:
                layer.lora = True
            if depth >-1 and i > depth:
                break
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            print(f'layer:{i}')
        visual_layer = x
        x = self.last_layer(x, adj)  # 最后一层不执行.detach()
        x = F.log_softmax(x, dim=1)
        return x, 0, 1, 2, visual_layer

class GraphConvolution_GCN(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(GraphConvolution_GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora = False
        self.A = Parameter(torch.FloatTensor(in_features, rank).uniform_(0.01, 0.02))
        self.B = Parameter(torch.FloatTensor(rank, out_features).uniform_(0.01, 0.02))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if in_features == out_features:
            self.weight = Parameter(torch.eye(in_features))
            self.reset_parameters()
            identity_matrix = torch.eye(self.weight.size(0))
            # data = self.weight.data
            # data = data * 0.1
            # data = torch.where(identity_matrix == 1, torch.tensor(1.0),data)
            self.weight.data = identity_matrix
            self.bias.data = torch.zeros(self.bias.shape)
        else:
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
            self.reset_parameters()
        #self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # 第一层(1433,256)；第二层(256,7)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv) # 第一层(1433,16) ，第二层(16,7)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = torch.spmm(adj, input) # 第一层(2708,2708),(2708,1433)->(2708,1433)；第二层(2708,2708),(2708,256)->(2708,256)
        if self.lora:
            weight = self.weight.detach() # 第一层(1433,16) * (16,7) -> (1433,7)
            bias = self.bias.detach()
            if self.rank > 0:
                weight+=torch.mm(self.A, self.B)
        else:
            weight = self.weight
            bias = self.bias
        support = torch.mm(input, weight)
        output = support
        if self.bias is not None:
            return output + bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'