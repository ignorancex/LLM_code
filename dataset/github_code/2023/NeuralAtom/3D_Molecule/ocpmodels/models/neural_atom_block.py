import torch
from einops import einsum, rearrange, reduce
from ocpmodels.models.gemnet.layers.base_layers import Dense, ResidualLayer
from ocpmodels.modules.scaling.scale_factor import ScaleFactor
from torch_geometric.utils import to_dense_batch

from .na_pooling import Exchanging, Porjecting


class NeuralAtom(torch.nn.Module):
    """
    Long-range block from the Neural Atom method

    Parameters
    ----------
        shared_downprojection: Dense,
            Downprojection block in Ewald block update function,
            shared between subsequent Ewald Blocks.
        emb_size_atom: int
            Embedding size of the atoms.
        downprojection_size: int
            Dimension of the downprojection bottleneck
        num_hidden: int
            Number of residual blocks in Ewald block update function.
        activation: callable/str
            Name of the activation function to use in the dense layers.
        scale_file: str
            Path to the json file containing the scaling factors.
        name: str
            String identifier for use in scaling file.
        use_pbc: bool
            Set to True if periodic boundary conditions are applied.
        delta_k: float
            Structure factor voxel resolution
            (only relevant if use_pbc == False).
        k_rbf_values: torch.Tensor
            Pre-evaluated values of Fourier space RBF
            (only relevant if use_pbc == False).
        return_k_params: bool = True,
            Whether to return k,x dot product and damping function values.
    """

    def __init__(
        self,
        emb_size_atom: int,
        num_hidden: int,
        activation=None,
        name=None,  # identifier in case a ScalingFactor is applied to Ewald output
    ):
        super().__init__()

        self.pre_residual = ResidualLayer(
            emb_size_atom, nLayers=2, activation=activation
        )

        # NOTE: using half size of interaction embedding size

        self.linear_heads = self.get_mlp(
            emb_size_atom, emb_size_atom, num_hidden, activation
        )
        if name is not None:
            self.ewald_scale_sum = ScaleFactor(name + "_sum")
        else:
            self.ewald_scale_sum = None
        number_atoms = 100
        ratio = 0.1

        self.proj_layer = Porjecting(
            emb_size_atom,
            num_heads=1,
            num_seeds=int(number_atoms * ratio),
            Conv=None,
            layer_norm=True,
        )
        self.interaction_layer = Exchanging(
            emb_size_atom,
            emb_size_atom,
            num_heads=2,
            Conv=None,
            layer_norm=True,
        )

    def get_mlp(self, units_in, units, num_hidden, activation):
        dense1 = Dense(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for i in range(num_hidden)
        ]
        mlp += res
        return torch.nn.ModuleList(mlp)

    '''
    h: embedding
    x: pos embedding
    batch_seg: batch index
    '''

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
    ):
        hres = self.pre_residual(h)

        # * use hres to perform message agg and passing
        h_update = self.get_NAs_emb(hres, batch_seg)

        # Apply update function
        for layer in self.linear_heads:
            h_update = layer(h_update)

        return h_update

    def get_NAs_emb(self, x, batch):
        batch_x, ori_mask = to_dense_batch(x, batch)
        mask = (~ori_mask).unsqueeze(1).to(dtype=x.dtype) * -1e9
        # * S for node cluster allocation matrix
        NAs_emb, S = self.proj_layer(batch_x, None, mask)
        NAs_emb = self.interaction_layer(NAs_emb, None, None)[0]
        h = reduce(
            einsum(
                rearrange(S, "(b h) c n -> h b c n", h=1),
                NAs_emb,
                "h b c n, b c d -> h b n d",
            ),
            "h b n d -> b n d",
            "mean",
        )[ori_mask]
        return h
