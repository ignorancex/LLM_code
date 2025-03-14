
import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.utils import spmm
from cogdl.models import BaseModel


class Grand(BaseModel):
    """
    Implementation of GRAND in paper `"Graph Random Neural Networks for Semi-Supervised Learning on Graphs"`
    <https://arxiv.org/abs/2005.11079>

    Parameters
    ----------
    nfeat : int
        Size of each input features.
    nhid : int
        Size of hidden features.
    nclass : int
        Number of output classes.
    input_droprate : float
        Dropout rate of input features.
    hidden_droprate : float
        Dropout rate of hidden features.
    use_bn : bool
        Using batch normalization.
    dropnode_rate : float
        Rate of dropping elements of input features
    tem : float
        Temperature to sharpen predictions.
    lam : float
         Proportion of consistency loss of unlabelled data
    order : int
        Order of adjacency matrix
    sample : int
        Number of augmentations for consistency loss
    alpha : float
    """
    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        input_droprate=0.5,
        hidden_droprate=0.5,
        use_bn=False,
        dropnode_rate=0.1,
        order=2,
        alpha=0.5,
    ):
        super(Grand, self).__init__()
        self.layer1 = nn.Linear(nfeat, nhid)
        self.layer2 = nn.Linear(nhid, nclass)
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.order = order
        self.dropnode_rate = dropnode_rate
        self.alpha = alpha

    def drop_node(self, x):
        n = x.shape[0]
        drop_rates = torch.ones(n) * self.dropnode_rate
        if self.training:
            masks = torch.bernoulli(1.0 - drop_rates).unsqueeze(1)
            x = masks.to(x.device) * x

        else:
            x = x * (1.0 - self.dropnode_rate)
        return x

    def reset_parameters(self):
        self.layer1.reset_parameters()  
        self.layer2.reset_parameters()  

    def rand_prop(self, graph, x):
        x = self.drop_node(x)

        y = x
        for i in range(self.order):
            x = spmm(graph, x).detach_()
            y.add_(x)
        return y.div_(self.order + 1.0).detach_()

    def normalize_x(self, x):
        row_sum = x.sum(1)
        row_inv = row_sum.pow_(-1)
        row_inv.masked_fill_(row_inv == float("inf"), 0)
        x = x * row_inv[:, None]
        return x

    def forward(self, graph):
        graph.sym_norm()
        x = graph.x
        x = self.normalize_x(x)
        x = self.rand_prop(graph, x)
        if self.use_bn:
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer2(x)
        return F.log_softmax(x, dim=-1)

    def predict(self, data):
        return self.forward(data)

