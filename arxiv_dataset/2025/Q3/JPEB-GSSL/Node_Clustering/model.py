from torch.nn import Module
from torch_geometric.nn import ChebConv, GCNConv, GraphNorm
import torch.nn.functional as F


class ContextEncoder(Module):
    def __init__(self, in_features):
        super(ContextEncoder, self).__init__()

        self.gcn1 = GCNConv(in_channels=in_features,
                            out_channels=128)
        self.gn1 = GraphNorm(128)
        self.gcn2 = GCNConv(in_channels=128,
                            out_channels=256)
        self.gn2 = GraphNorm(256)
        self.gcn3 = GCNConv(in_channels=256,
                            out_channels=512)
        self.gn3 = GraphNorm(512)

    def forward(self, x, edge_index):

        x = self.gcn1(x, edge_index=edge_index)
        x = F.relu(self.gn1(x))

        x = self.gcn2(x, edge_index=edge_index)
        x = F.relu(self.gn2(x))

        x = self.gcn3(x, edge_index=edge_index)
        x = F.tanh(self.gn3(x))

        return x
