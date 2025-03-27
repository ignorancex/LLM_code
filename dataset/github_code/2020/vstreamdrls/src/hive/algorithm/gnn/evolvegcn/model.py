from torch.nn.parameter import Parameter
from torch.nn.modules.module import  Module
import torch.nn as nn
import torch
import math

class EvolveGCNModel(Module):
    def __init__(self, layer_1_in, layer_1_out, layer_2_out):
        super(EvolveGCNModel, self).__init__()
        self.layer_1_grcu = GRCU(layer_1_in, layer_1_out, torch.nn.RReLU())
        self.layer_2_grcu = GRCU(layer_1_out, layer_2_out, torch.nn.RReLU())


    def forward(self, adj_list, features):
        new_features = self.layer_1_grcu(adj_list, features)
        node_emb = self.layer_2_grcu(adj_list, new_features)
        return node_emb[-1]

    def __str__(self):
        return "EVOLVEGCN"



class GRCU(Module):
    def __init__(self, layer_in, layer_out, activation):
        super(GRCU, self).__init__()


        self.GCN_init_weights = Parameter(torch.Tensor(layer_in, layer_out))
        nn.init.xavier_uniform_(self.GCN_init_weights)
        self.activation = activation

        self.evolve_weights = mat_GRU_cell(layer_in, layer_out)

    def forward(self, adj_list, features_list):
        gcn_weights = self.GCN_init_weights
        out_seq = []
        for i, adj in enumerate(adj_list):
            features = features_list[i]
            gcn_weights = self.evolve_weights(gcn_weights)
            embs = self.activation(torch.spmm(adj, torch.mm(features, gcn_weights)))
            out_seq.append(embs)
        return out_seq


class mat_GRU_cell(Module):
    def __init__(self, rows, cols):
        super(mat_GRU_cell, self).__init__()
        self.update = mat_GRU_gate(rows,
                                   cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(rows,
                                  cols,
                                  torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(rows,
                                   cols,
                                   torch.nn.Tanh())

        self.choose_topk = TopK(feats=rows,
                                k=cols)

    def forward(self, prev_Q):  # ,prev_Z,mask):
        # z_topk = self.choose_topk(prev_Z,mask)
        z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class mat_GRU_gate(Module):
    def __init__(self, rows, cols, activation):
        super(mat_GRU_gate, self).__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows, cols))

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out


class TopK(Module):
    def __init__(self, feats, k):
        super(TopK, self).__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        # Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs, mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
                isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        # we need to transpose the output
        return out.t()


def pad_with_last_val(vect,k):
    device = 'cuda' if vect.is_cuda else 'cpu'
    pad = torch.ones(k - vect.size(0),
                         dtype=torch.long,
                         device = device) * vect[-1]
    vect = torch.cat([vect,pad])
    return vect