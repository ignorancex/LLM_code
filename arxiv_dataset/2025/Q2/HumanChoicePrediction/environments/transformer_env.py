from environments import environment
import torch
import torch.nn as nn
from SpecialLSTM import SpecialLSTM
from consts import *


class transformer_env_ARC(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config["input_dim"]
        dropout = config["dropout"]
        nhead = config["transformer_nheads"]
        hidden_dim = config["hidden_dim"]
        self.user_vectors = None

        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                ).double()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.main_task = nn.TransformerEncoder(self.encoder_layer, num_layers=config["layers"]).double()

        self.main_task_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                                  nn.ReLU(),
                                                  nn.Linear(hidden_dim // 2, 2),
                                                  nn.LogSoftmax(dim=-1)).double()


    def forward(self, vectors, **kwargs):
        x = vectors["x"]
        x = self.fc(x)
        output = []
        for i in range(DATA_ROUNDS_PER_GAME):
            time_output = self.main_task(x[:, :i+1].contiguous())[:, -1, :]
            output.append(time_output)
        output = torch.stack(output, 1)
        output = self.main_task_classifier(output)
        return {"output": output}

    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        assert not update_vectors
        output = self(data)
        output["proba"] = torch.exp(output["output"].flatten())
        return output


class transformer_env(environment.Environment):
    def init_model_arc(self, config):
        self.model = transformer_env_ARC(config=config).double()

    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        if vectors_in_input:
            output = self.model(data)
        else:
            raise NotImplementedError
            # output = self.model({**data, "user_vector": self.currentDM, "game_vector": self.currentGame})
        output["proba"] = torch.exp(output["output"].flatten())
        # if update_vectors:
        #     self.currentDM = output["user_vector"]
        #     self.currentGame = output["game_vector"]
        return output


    def init_user_vector(self):
        self.currentDM = self.model.init_user()

    def init_game_vector(self):
        self.currentGame = self.model.init_game()

    def get_curr_vectors(self):
        return {"user_vector": 888, }