import rootutils
import torch
import torch.nn as nn

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.KP.models import MLP, MLP_LSTM_FE
from src.models.modules import FeatureMapExtractorModel, RNNHead


class Encoder(nn.Module):
    def __init__(self, fe: nn.Module, MLP: nn.Module):
        super().__init__()
        self.fe = fe
        self.mlp = MLP

    def forward(self, x):
        x = self.fe(x)
        x = self.mlp(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        rnn_head: nn.Module,
    ):
        super().__init__()
        self.rnn_head = rnn_head

    def forward(self, x):
        return self.rnn_head(x)


def load_model(path_to_weights, cfg):
    fe = FeatureMapExtractorModel(num_keypoints=cfg.fe.num_keypoints, out_dim=cfg.fe.out_dim)
    mlp = MLP(
        input_dim=cfg.mlp.input_dim,
        hidden_dim=cfg.mlp.hidden_dim,
        output_dim=cfg.mlp.output_dim,
    )
    rnn_head = RNNHead(
        input_dim=cfg.decoder.input_dim,
        hidden_dim=cfg.decoder.hidden_dim,
        num_layers=cfg.decoder.num_layers,
        bidirectional=cfg.decoder.bidirectional,
        return_outs=cfg.decoder.return_outs,
        num_classes=cfg.decoder.num_classes,
    )
    model = MLP_LSTM_FE(fe=fe, mlp=mlp, decoder=rnn_head)

    model.load_state_dict(torch.load(path_to_weights, map_location=torch.device("cpu")))
    encoder = Encoder(fe=fe, MLP=mlp)
    decoder = Decoder(rnn_head=rnn_head)
    return encoder, decoder
