from torch.nn import Tanh, Module, ReLU
from torch_geometric.nn import GCNConv, GraphNorm
import torch.nn.functional as F


class ContextTargetPredictor(Module):
    def __init__(self, dims):
        super(ContextTargetPredictor, self).__init__()

        self.gc1 = GCNConv(
            in_channels=dims, out_channels=dims, normalize=False)
        self.norm = GraphNorm(dims)
        self.tanh1 = Tanh()

        self.gc2 = GCNConv(out_channels=dims,
                           in_channels=dims, normalize=False)
        self.tanh2 = Tanh()

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index=edge_index)
        x = self.norm(x)
        x = self.tanh1(x)

        x = self.gc2(x, edge_index=edge_index)
        x = self.tanh2(x)

        return x
