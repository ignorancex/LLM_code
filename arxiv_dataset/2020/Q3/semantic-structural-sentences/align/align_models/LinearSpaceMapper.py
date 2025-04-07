import torch
import torch.nn as nn

from .BaseModel import BaseModel


class LinearSpaceMapper(BaseModel):


    def __init__(self, _sent_weights, _entity_weights, _relation_weights, _concat, _normalize_vecs, _beta, _tune):
        super().__init__()
        # torch.manual_seed(17)
        self.concat_method = _concat
        self.sent_weights = _sent_weights
        self.ent_weights = _entity_weights
        self.rel_weights = _relation_weights
        self.normalize_vecs = _normalize_vecs
        self.normalize_embeddings()
        self.beta = _beta
        self.update_embeddings = _tune

        self.sent_embedding, self.sent_dim = self.create_emb_layer(self.sent_weights, _tune)
        self.entity_embedding, self.ent_dim = self.create_emb_layer(self.ent_weights, _tune)
        self.rel_embedding, self.rel_dim = self.create_emb_layer(self.rel_weights, _tune)
        self.kg_dim = self.compute_kg_dim()
        self.mapping = nn.Linear(self.kg_dim, self.sent_dim, bias=False)
        torch.nn.init.orthogonal_(self.mapping.weight)

    def forward(self, _n_in):
        triples = self.represent_triple(_n_in[:, 0], _n_in[:, 1], _n_in[:, 2])
        out = self.mapping(triples)
        out_norm = nn.functional.normalize(out, p=2, dim=1)
        return out_norm

    def prior_space(self, _n_in):
        triples = self.represent_triple(_n_in[:, 0], _n_in[:, 1], _n_in[:, 2])
        return triples

    def orthogonalize(self):
        if self.beta > 0:
            W = self.mapping.weight.data
            beta = self.beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))


class ConstrainedLinearSpaceMapper(BaseModel):

    def __init__(self, _node_weights, _sent_weights):
        super().__init__()
        self.init_nodes = _node_weights
        self.init_sents = _sent_weights
        self.node_embedding, self.node_dim = create_emb_layer(self.init_nodes, True)
        self.sent_embedding, self.sent_dim = create_emb_layer(self.init_sents, True)
        # TODO: need to make sure the mapping is orthonormal
        # TODO: should this be constrained or unconstrained? if constrained, what should the init weights
        # be and should some orthogonality also be required?
        # if unconstrained, should early stopping be driven by the dev set to prevent overfitting
        # nn.init.uniform_(self.mapping.weight, -1.0, 1.0)
        # TODO: https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
        self.mapping = nn.Linear(self.node_dim, self.sent_dim)

    def forward(self, _n_in):
        forw = self.mapping(self.node_embedding(_n_in))
        return nn.init.orthogonal_(forw, gain=1)
