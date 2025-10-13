import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from wyckoff_generation.common.registry import registry
from wyckoff_generation.models import mlp


class WyckoffGNNLayer(MessagePassing):
    def __init__(
        self,
        node_hidden_dim,
        dof_emb_dim,
        hidden_dim,
        num_hidden_layers,
        activation,
        no_softmax,
    ):
        super().__init__("sum")
        # TODO: implement true Transformer attention?
        self.a_mlp = mlp.get_mlp(
            2 * (node_hidden_dim + dof_emb_dim),
            1,
            hidden_dim,
            num_hidden_layers,
            activation,
        )
        # for backward compability
        if no_softmax:
            self.a_out_fn = lambda h, index: h
        else:
            self.a_out_fn = softmax

        self.psi_mlp = mlp.get_mlp(
            node_hidden_dim + dof_emb_dim,
            node_hidden_dim,
            hidden_dim,
            num_hidden_layers,
            activation,
        )

    def message(self, h_i, h_j, index):
        z = torch.cat([h_i, h_j], dim=1)
        a = self.a_mlp(z)
        # index is automatically set to the indices of the central nodes (i)
        # this is used to group the a-values, then softmax is applied over those elements (over j)
        a = self.a_out_fn(a, index)
        psi_i = self.psi_mlp(h_j)
        return a * psi_i

    def forward(self, h, h_dof, edge_index):
        z = torch.cat([h, h_dof], dim=1)
        out = self.propagate(edge_index, h=z)
        return h + out


@registry.register_gnn("base")
class WyckoffGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.max_num_atoms = config[
            "max_num_atoms"
        ]  # max number of atoms of each specie
        self.max_atom_num = config["num_elements"]
        self.layers = nn.ModuleList(
            [
                WyckoffGNNLayer(
                    config["hidden_dim"],
                    config["dof_pos_sg_emb_size"],
                    2 * (config["hidden_dim"] + config["dof_pos_sg_emb_size"]),
                    config["mlp_hidden_layers"],
                    config["mlp_activation"],
                    config.get("no_softmax", True),
                )
                for _ in range(config["num_gnn_layers"])
            ]
        )
        self.activation = getattr(nn, config["gnn_activation"])()

        self.inf_dof_embedding = nn.Embedding(self.max_num_atoms + 1, 1)
        self.inf_dof_linear = nn.Linear(self.max_atom_num, config["hidden_dim"])
        self.zero_dof_embedding = nn.Embedding(
            self.max_atom_num + 1, config["hidden_dim"]
        )  # + 1 to include 0
        self.dof_embedding = nn.Embedding(
            2 if config.get("binary_dof_encoding", True) else 4,
            config["dof_pos_sg_emb_size"],
        )  # for backward compability

        self.pos_embedding = nn.Embedding(27, config["dof_pos_sg_emb_size"])

        self.sg_embedding = nn.Embedding(231, config["dof_pos_sg_emb_size"])

        self.time_embedding = nn.Embedding(
            config["t_max"], config["dof_pos_sg_emb_size"]
        )

        if config.get(
            "no_multiplicity_encoding", True
        ):  # if statement for backward compability
            self.multiplicity_embedding = None
        else:
            self.multiplicity_embedding = nn.Embedding(
                193, config["dof_pos_sg_emb_size"]
            )

        self.zero_df_out_mlp = mlp.get_mlp(
            config["hidden_dim"],
            self.max_atom_num + 1,
            2 * config["hidden_dim"],
            config["mlp_hidden_layers"],
            config["mlp_activation"],
        )
        self.inf_df_out_mlp = mlp.get_mlp(
            config["hidden_dim"],
            self.max_atom_num * (self.max_num_atoms + 1),
            2 * config["hidden_dim"],
            config["mlp_hidden_layers"],
            config["mlp_activation"],
        )

    def forward(self, data, t):
        edge_index = data.edge_index
        zero_dof = data.zero_dof

        zero_dof_h = self.zero_dof_embedding(data.x_0_dof)
        inf_dof_h = self.inf_dof_linear(
            self.inf_dof_embedding(data.x_inf_dof).squeeze()
        )
        h = torch.empty(
            (data.x_inf_dof.shape[0] + data.x_0_dof.shape[0], zero_dof_h.shape[1]),
            device=self.device,
        )
        h[zero_dof] = zero_dof_h
        h[~zero_dof] = inf_dof_h

        h_dof_pos_sg = (
            (
                self.dof_embedding(zero_dof.long())
                if self.dof_embedding.weight.shape[0] == 2
                else self.dof_embedding(data.degrees_of_freedom)
            )
            + self.pos_embedding(data.wyckoff_pos_idx)
            + self.sg_embedding(data.space_group).repeat_interleave(data.num_pos, dim=0)
            + self.time_embedding(t).repeat_interleave(data.num_pos, dim=0)
            + (
                self.multiplicity_embedding(data.multiplicities)
                if self.multiplicity_embedding is not None
                else 0.0
            )
        )

        for layer in self.layers:
            h = layer(h, h_dof_pos_sg, edge_index)
            h = self.activation(h)

        out_zero_df = self.zero_df_out_mlp(h[zero_dof])
        out_inf_df = self.inf_df_out_mlp(h[~zero_dof]).unflatten(
            1, (-1, self.max_num_atoms + 1)
        )
        return (
            out_zero_df,
            out_inf_df,
        )  # (num_0_dof, num_atom_types + 1) (num_inf_dof, num_atom_types, max_num_atoms + 1)
