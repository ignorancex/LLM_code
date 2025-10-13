from environments import environment
import torch
import torch.nn as nn
from SpecialLSTM import SpecialLSTM
from consts import *


class LSTM_env_ARC(SpecialLSTM):
    def forward(self, vectors, **kwargs):
        data = super().forward(vectors["x"], vectors["game_vector"], vectors["user_vector"])
        return data

    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        assert not update_vectors
        if vectors_in_input:
            output = self(data)
        else:
            output = self({**data, "user_vector": self.currentDM, "game_vector": self.currentGame})
        output["proba"] = torch.exp(output["output"].flatten())
        return output



class LSTM_env(environment.Environment):
    def init_model_arc(self, config):
        self.model = LSTM_env_ARC(n_layers=self.n_layers, input_dim=config['input_dim'], hidden_dim=self.hidden_dim,
                                  output_dim=config["output_dim"], dropout=config["dropout"]).double()

    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        if vectors_in_input:
            output = self.model(data)
        else:
            output = self.model({**data, "user_vector": self.currentDM, "game_vector": self.currentGame})
        output["proba"] = torch.exp(output["output"].flatten())
        if update_vectors:
            self.currentDM = output["user_vector"]
            self.currentGame = output["game_vector"]
        return output


    def init_user_vector(self):
        self.currentDM = self.model.init_user()

    def init_game_vector(self):
        self.currentGame = self.model.init_game()

    def get_curr_vectors(self):
        return {"user_vector": 888, }