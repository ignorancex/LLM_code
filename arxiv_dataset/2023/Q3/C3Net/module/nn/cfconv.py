import torch
from torch import nn
from module.nn.base import Aggregate, Dense


class CFConv(nn.Module):
    def __init__(self, n_in, n_filters, n_out, filter_network, activation=None, axis=2):
        super(CFConv, self).__init__()
        self.n_in = n_in
        self.in2f = Dense(n_in, n_filters, bias=False, activation=None)
        self.f2out = Dense(n_filters, n_out, bias=True, activation=activation)
        self.filter_network = filter_network
        self.agg = Aggregate(axis=axis)
 
    def forward(self, x, pairwise_mask, neighbors, f_ij=None):
        W = self.filter_network(f_ij) 
        
        if neighbors != None: #bond 
            nbh_size = neighbors.size()
            nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
            nbh = nbh.expand(-1, -1, x.size(2))
            x = torch.gather(x, 1, nbh)
            x = x.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)
        
        else:#solute-env
            x = torch.unsqueeze(x,1)
        
        # element-wise multiplication, aggregating and Dense layer
        y = self.in2f(x)
        y = y * W
    
        y = self.agg(y, pairwise_mask)
        y = self.f2out(y) 
        
        return y
