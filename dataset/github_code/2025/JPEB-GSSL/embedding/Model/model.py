from Model.context_encoder import ContextEncoder
from Model.predictor import ContextTargetPredictor
from torch_geometric.utils import dropout_node, k_hop_subgraph
from torch_geometric.nn import global_mean_pool
from torch.nn import Module
import torch


class EmbeddingModel(Module):
    def __init__(self, num_features, num_targets):
        super(EmbeddingModel, self).__init__()

        self.context_model = ContextEncoder(in_features=num_features)
        self.predictor_model = ContextTargetPredictor(dims=512)
        self.num_targets = num_targets

    def forward(self, G):
        # Consider a context subgraph
        edge_index, _, _ = dropout_node(
            G.edge_index, p=0.3)  # Bernoulli Distribution
        x = self.context_model(G.x, edge_index)
        e_u = []
        for _ in range(self.num_targets):
            v = self.predictor_model(x, edge_index)
            v_graph = global_mean_pool(v, batch=G.batch)
            e_u.append(v_graph)

        return e_u, x
