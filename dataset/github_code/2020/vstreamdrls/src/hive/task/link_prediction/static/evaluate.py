from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPEvaluation(Module):
    def __init__(self, emb_size):
        super(MLPEvaluation, self).__init__()
        self.W0 = Parameter(torch.zeros(size=(emb_size, emb_size), requires_grad=True))
        self.W1 = Parameter(torch.zeros(size=(emb_size, 1), requires_grad=True))
        nn.init.xavier_uniform_(self.W0)
        nn.init.xavier_uniform_(self.W1)

    def forward(self, inputs):
        x = F.relu(torch.matmul(inputs, self.W0))
        x = torch.sigmoid(torch.matmul(x, self.W1))
        return x


class Evaluation(object):
    def __init__(self, representations, train_edges_indices, train_edges_values, device):
        self.representations = representations
        self.train_edges_indices = train_edges_indices
        self.train_edges_values = train_edges_values
        self.evaluation_model = MLPEvaluation(self.representations.shape[1]).to(device=device)

    def train_evaluation_model(self):
        edge_representations = self.__get_edge_representations__(self.train_edges_indices)

        optimizer = optim.Adam(self.evaluation_model.parameters(), lr=1e-3)
        for epoch in range(100):
            self.evaluation_model.train()
            optimizer.zero_grad()
            output = self.evaluation_model(edge_representations)
            criterion = nn.MSELoss()
            loss = criterion(output.squeeze(), self.train_edges_values)
            loss.backward()
            optimizer.step()
        self.evaluation_model.eval()


    def evaluate(self, indices, values):
        edge_representations = self.__get_edge_representations__(indices)
        output = self.evaluation_model(edge_representations)
        rmse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        mae_score = mae_criterion(output.squeeze(), values)
        rmse_score = torch.sqrt(rmse_criterion(output.squeeze(), values))
        return mae_score.item(), rmse_score.item()


    def __get_edge_representations__(self, indices):
        src_representations = torch.index_select(self.representations, 0, indices[0])
        dst_representations = torch.index_select(self.representations, 0, indices[1])
        return torch.mul(src_representations, dst_representations)