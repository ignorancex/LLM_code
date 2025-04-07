import torch
import torch.nn as nn

from .BaseModel import BaseModel


class NonLinearSpaceMapper(BaseModel):

    def __init__(self, _sent_weights, _entity_weights, _relation_weights,
                 _concat, _normalize_vecs, _tune, _hidden_size, _dropout):
        super().__init__()
        torch.manual_seed(17)
        self.concat_method = _concat
        self.sent_weights = _sent_weights
        self.ent_weights = _entity_weights
        self.rel_weights = _relation_weights
        self.normalize_vecs = _normalize_vecs
        self.normalize_embeddings()
        self.update_embeddings = _tune

        self.sent_embedding, self.sent_dim = self.create_emb_layer(self.sent_weights, _tune)
        self.entity_embedding, self.ent_dim = self.create_emb_layer(self.ent_weights, _tune)
        self.rel_embedding, self.rel_dim = self.create_emb_layer(self.rel_weights, _tune)
        self.kg_dim = self.compute_kg_dim()

        self.hidden_size = _hidden_size
        self.dropout = _dropout

        self.h1 = nn.Linear(self.kg_dim, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.d1 = nn.Dropout(self.dropout)
        self.h2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()
        self.d2 = nn.Dropout(self.dropout)
        self.h3 = nn.Linear(self.hidden_size, self.sent_dim)

    def forward(self, _n_in):
        triples = self.represent_triple(_n_in[:, 0], _n_in[:, 1], _n_in[:, 2])
        x = self.h1(triples)
        x = self.relu1(x)
        x = self.d1(x)
        x = self.h2(x)
        x = self.relu2(x)
        x = self.d2(x)
        x = self.h3(x)
        out_norm = nn.functional.normalize(x, p=2, dim=1)
        return out_norm

    def prior_space(self, _n_in):
        triples = self.represent_triple(_n_in[:, 0], _n_in[:, 1], _n_in[:, 2])
        return triples
