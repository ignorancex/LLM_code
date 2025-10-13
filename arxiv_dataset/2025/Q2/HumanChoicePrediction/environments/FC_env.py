from environments import environment
import torch
import torch.nn as nn
from consts import *

class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc11 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Activation function
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, vectors):
        x = vectors["x"]
        shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc11(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.logsoftmax(x)
        x.reshape(shape[:-1] + (2,))
        return {"output": x}


class FC_env(environment.Environment):
    def init_model_arc(self, config):
        input_dim = config["REVIEW_DIM"] + STRATEGY_DIM  # depends on the size of your input
        hidden_dim = 128  # can be any number
        output_dim = 2  # depends on the size of your output
        self.model = FullyConnectedNN(input_dim, hidden_dim, output_dim)
        self.model.eval()

    def predict_proba(self, x):
        return {"proba": torch.exp(self.model(x))}
