from torch.nn import Linear, Sequential, GELU
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops
import torch
import torch.nn.functional as F


class EGNN(GATv2Conv):
    def __init__(
            self,
            in_channels,
            out_channels,
            heads=6,
            dropout=0.2,
            edge_feature_dims=10,
            edge_coords_nf=1,
            activation=GELU()):
        super().__init__(in_channels, out_channels, heads, dropout, edge_dim=(edge_feature_dims + edge_coords_nf))

        self.coord_mlp = Sequential(
            activation,
            Linear(out_channels, 1).double()
        )

        self.lin = Linear(in_channels, heads * out_channels)

    @staticmethod
    def coord2radial(edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
        return radial, coord_diff

    @staticmethod
    def unsorted_segment_sum(data, segment_ids, num_segments):
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)  # Init empty result tensor.
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)
        return result

    def forward(self, h, edge_index, coord, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index, num_nodes=h.size(0), edge_attr=edge_attr, fill_value='mean')

        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_attr = torch.cat([edge_attr, coord_diff], dim=-1)

        h = self.lin(h.float()).view(-1, self.heads, self.out_channels)

        hidden_out, coors_out = self.propagate(edge_index, x=h.float(), edge_attr=edge_attr.float(), coors=coord.float(),
                                               rel_coors=coord_diff.float())

        return F.relu(hidden_out), coors_out

    def propagate(self, edge_index, size=None, **kwargs):

        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)

        m_ij = self.message(**msg_kwargs).view(-1, self.heads * self.out_channels)

        coor_wij = self.coord_mlp(m_ij.view(-1, self.heads, self.out_channels)[:, 0, :])

        mhat_i = self.aggregate(coor_wij * kwargs["rel_coors"], **aggr_kwargs)
        coors_out = kwargs["coors"] + mhat_i

        m_i = self.aggregate(m_ij, **aggr_kwargs).view(-1, self.heads*self.out_channels)

        # return tuple
        return self.update((m_i, coors_out), **update_kwargs)