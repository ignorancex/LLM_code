import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .Model import Model


class ShallowGRU(Model):
    """
    Shallow GRU model for AG News dataset
    """

    def __init__(self, embedding, shape_out=4):
        super(ShallowGRU, self).__init__()

        embedding_dim = embedding.shape[1]

        self.embed = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=32, num_layers=1, bidirectional=False,
                          batch_first=True)
        self.linear = nn.Linear(32, shape_out)

        self.embed_weight = None

    def forward(self, x, lens):
        x = self.embed(x)
        x = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x)
        x, lens = pad_packed_sequence(x, batch_first=True)

        # global pooling
        x = x.sum(dim=1) / lens.view((-1, 1)).to(x.device)
        x = self.linear(x)
        return x

    def uploaded_state_dict(self):
        state = self.state_dict()
        state.pop('embed.weight')  # since embed weight is not trained, no need to update it
        return state


def test():
    model = ShallowGRU(embedding=torch.randn((200, 100)), shape_out=4)
    print(model.uploaded_state_dict().keys())
    print(model.get_params_tensor().shape)
