# 3p
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Dropout, LeakyReLU, Linear
from torch_geometric.nn import global_max_pool, knn_graph
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter
from deltaconv.nn import MLP, VectorMLP
from deltaconv.geometry.grad_div_mls import build_grad_div, build_tangent_basis, estimate_basis
from deltaconv.geometry.operators import curl, norm, I_J, hodge_laplacian


class ASAPDeltaConv(torch.nn.Module):
    def __init__(self, dim_in=3, dim_out=128, dropout=0.5, neig=128, conv_channels=[64, 128, 256], mlp_depth=2,
                 embedding_size=1024, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1):
        super().__init__()

        self.neig = neig

        self.deltaconv_net = DeltaNetSegmentation(dim_in, dim_out, conv_channels, mlp_depth,
                                                  embedding_size, num_neighbors, grad_regularizer, grad_kernel_width, dropout)

    def forward(self, x, pos, norm, mass, evals, evecs):
        # input data
        evecs, evals = evecs[:, :, :self.neig], evals[:, :self.neig]
        mass = mass.float()
        evecs_trans = torch.bmm(evecs.transpose(2, 1), torch.diag_embed(mass))

        # create a new batch
        batch_list = [Data(x=x[i], pos=pos[i], norm=norm[i]) for i in range(x.size(0))]
        pyg_batch = Batch.from_data_list(batch_list)
        pyg_batch.evecs = evecs
        pyg_batch.evals = evals
        pyg_batch.evecs_trans = evecs_trans

        # forward pass
        features = self.deltaconv_net(pyg_batch)

        # reshape features using batch information
        features = to_dense_batch(features, pyg_batch.batch)[0]

        # output data
        return features


class DeltaNetSegmentation(torch.nn.Module):
    def __init__(self, in_channels, num_classes, conv_channels=[64, 128, 256], mlp_depth=2,
                 embedding_size=1024, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1, dropout=0.5):
        """Segmentation of Point Clouds with DeltaConv.
        The architecture is based on the architecture used by DGCNN (https://dl.acm.org/doi/10.1145/3326362.

        Args:
            in_channels (int): the number of channels provided as input.
            num_classes (int): the number of classes to segment.
            conv_channels (list[int]): the number of output channels of each convolution.
            mlp_depth (int): the depth of the MLPs of each convolution.
            embedding_size (int): the embedding size before the segmentation head is applied.
            num_neighbors (int): the number of neighbors to use in estimating the gradient.
            grad_regularizer (float): the regularizer value used in the least-squares fitting procedure.
                In the paper, this value is referred to as \lambda.
                Larger grad_regularizer gives a smoother, but less accurate gradient.
                Lower grad_regularizer gives a more accurate, but more variable gradient.
                The grad_regularizer value should be >0 (e.g., 1e-4) to prevent exploding values.
            grad_kernel_width (float): the width of the gaussian kernel used to weight the
                least-squares problem to approximate the gradient.
                Larger kernel width means that more points are included, which is a 'smoother' gradient.
                Lower kernel width gives a more accurate, but possibly noisier gradient.
        """
        super().__init__()

        self.deltanet_base = DeltaNetBase(in_channels, conv_channels, mlp_depth, num_neighbors, grad_regularizer,
                                          grad_kernel_width)

        # Global embedding
        feat_new_size = sum(conv_channels) * 2
        self.lin_global = MLP([feat_new_size, embedding_size])

        # For ShapeNet segmentation, most authors add an embedding of the category to aid with segmentation.
        self.segmentation_head = Seq(
            MLP([embedding_size + feat_new_size, 256]), Dropout(dropout), MLP([256, 256]), Dropout(dropout),
            Linear(256, 128), LeakyReLU(negative_slope=0.2), Linear(128, num_classes))

    def forward(self, data):
        conv_out = self.deltanet_base(data)

        x = torch.cat(conv_out, dim=1)
        x = self.lin_global(x)

        batch = data.batch
        x_max = global_max_pool(x, batch)[batch]

        x = torch.cat([x_max] + conv_out, dim=1)

        return self.segmentation_head(x)


class DeltaNetBase(torch.nn.Module):
    def __init__(self, in_channels, conv_channels, mlp_depth, num_neighbors, grad_regularizer, grad_kernel_width, centralize_first=True):
        """Classification of Point Clouds with DeltaConv.
        The architecture is based on the architecture used by DGCNN (https://dl.acm.org/doi/10.1145/3326362.

        Args:
            in_channels (int): the number of channels provided as input.
            conv_channels (list[int]): the number of output channels of each convolution.
            mlp_depth (int): the depth of the MLPs of each convolution.
            num_neighbors (int): the number of neighbors to use in estimating the gradient.
            grad_regularizer (float): the regularizer value used in the least-squares fitting procedure.
                In the paper, this value is referred to as \lambda.
                Larger grad_regularizer gives a smoother, but less accurate gradient.
                Lower grad_regularizer gives a more accurate, but more variable gradient.
                The grad_regularizer value should be >0 (e.g., 1e-4) to prevent exploding values.
            grad_kernel_width (float): the width of the gaussian kernel used to weight the
                least-squares problem to approximate the gradient.
                Larger kernel width means that more points are included, which is a 'smoother' gradient.
                Lower kernel width gives a more accurate, but possibly noisier gradient.
            centralize_first (bool, optional): whether to centralize the input features (default: True).
        """
        super().__init__()
        self.k = num_neighbors
        self.grad_regularizer = grad_regularizer
        self.grad_kernel_width = grad_kernel_width

        # Create convolution layers
        conv_channels = [in_channels] + conv_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append(DeltaConv(conv_channels[0], conv_channels[1], depth=mlp_depth, centralized=(centralize_first and 0 == 0),
                                    vector=True))
        for i in range(1, len(conv_channels) - 1):
            last_layer = i == (len(conv_channels) - 2)
            self.convs.append(DeltaConv(conv_channels[i] * 2, conv_channels[i + 1], depth=mlp_depth, centralized=(centralize_first and i == 0),
                                        vector=not (last_layer)))

        self.specs = torch.nn.ModuleList()
        for i in range(len(conv_channels) - 1):
            self.specs.append(LaplacianBlock(conv_channels[i + 1], False))

    def forward(self, data):
        pos = data.pos
        batch = data.batch

        # Operator construction
        # ---------------------

        # Create a kNN graph, which is used to:
        # 1) Perform maximum aggregation in the scalar stream.
        # 2) Approximate the gradient and divergence oeprators
        edge_index = knn_graph(pos, self.k, batch, loop=True, flow='target_to_source')

        # Use the normals provided by the data or estimate a normal from the data.
        #   It is advised to estimate normals as a pre-transform.

        # Note: the x_basis and y_basis are referred to in the DeltaConv paper as e_u, and e_v, respectively.
        # Wherever x and y are used to denote tangential coordinates, they can be interchanged with u and v.
        if hasattr(data, 'norm') and data.norm is not None:
            normal = data.norm
            x_basis, y_basis = build_tangent_basis(normal)
        else:
            edge_index_normal = knn_graph(pos, 10, batch, loop=True, flow='target_to_source')
            # When normal orientation is unknown, we opt for a locally consistent orientation.
            normal, x_basis, y_basis = estimate_basis(pos, edge_index_normal, orientation=pos)

        # Build the gradient and divergence operators.
        # grad and div are two sparse matrices in the form of SparseTensor.
        grad, div = build_grad_div(pos, normal, x_basis, y_basis, edge_index, batch, kernel_width=self.grad_kernel_width, regularizer=self.grad_regularizer)

        # Forward pass convolutions
        # ---------------------------------

        # The scalar features are stored in x
        x = data.x if hasattr(data, 'x') and data.x is not None else pos
        # Vector features in v
        v = grad @ x

        # Store each of the interim outputs in a list
        out = []
        for ind, conv in enumerate(self.convs):
            x, v = conv(x, v, grad, div, edge_index)
            # smooth the features
            x_dense = to_dense_batch(x, data.batch)[0]
            x_projected = torch.bmm(data.evecs_trans, x_dense)
            x_smooth = torch.bmm(data.evecs, self.specs[ind](x_projected, data.evals))
            x = torch.cat((x_dense, x_smooth), dim=-1)
            x = x.reshape(-1, x.size(-1))
            out.append(x)

        # Return the interim outputs
        return out


class DeltaConv(torch.nn.Module):
    """ DeltaConv convolution from the paper
    "DeltaConv: Anisotropic Operators for Geometric Deep Learning on Point Clouds".
    This convolution learns a combination of operators from vector calculus:
        grad, co-grad, div, curl; and their compositions Laplacian and Hodge-Laplacian
    and separates features into a scalar and vector stream.

    DeltaConv can be applied to any discretization. Simply provide the discretized gradient and divergence
    Depending on the discretization, the implementation of the rotation matrix (J) and norm should be updated.

    Args:
        in_channels (int): the number of input channels of the features.
        out_channels (int): the number of output channels after the convolution.
        depth (int, optional): the depth of the MLPs (default: 1).
        centralized (bool, optional): centralizes the input features
            before maximum aggregation if set to True (default: False):
            p_j = p_j - p_i.
        vector (bool, optional): determines whether the vector stream is propagated 
            set this to false in the last layer of a network that only outputs scalars (default: True).
        aggr (string, optional): the type of aggregation used in the scalar stream (default: 'max').
    """
    def __init__(self, in_channels, out_channels, depth=1, centralized=False, vector=True, aggr='max'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.centralized = centralized
        self.aggr = aggr

        self.s_mlp_max = MLP([in_channels] + [out_channels] * depth)
        self.s_mlp = MLP([in_channels * 4] + [out_channels] * depth)
        if vector:
            self.v_mlp = VectorMLP([in_channels * 4 + out_channels * 2] + [out_channels * 2] * depth)
        else:
            self.v_mlp = None

    def forward(self, x, v, grad, div, edge_index):

        # Scalar -> Scalar, Vector -> Scalar
        # ----------------------------------

        # Aggregation in scalar stream, defaults to maximum aggregation.
        if self.centralized:
            x_edge = x[edge_index[1]] - x[edge_index[0]]
            x_max = scatter(self.s_mlp_max(x_edge), edge_index[0], dim=0, reduce=self.aggr)
        else:
            x_max = scatter(self.s_mlp_max(x)[edge_index[1]], edge_index[0], dim=0, reduce=self.aggr)

        # Apply operators and concatenate.
        x_cat = torch.cat([x, div @ v, curl(v, div), norm(v)], dim=1)
        # Combine the operators with an MLP.
        x = x_max + self.s_mlp(x_cat)

        # Vector -> Vector, Scalar -> Vector
        # ----------------------------------

        if self.v_mlp is not None:
            # Apply operators and concatenate.
            v_cat = torch.cat([v, hodge_laplacian(v, grad, div), grad @ x], dim=1)
            # Combine the operators and their 90-degree rotated variants (I_J) with an MLP.
            v = self.v_mlp(I_J(v_cat))

        return x, v

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'


class LaplacianBlock(nn.Module):
    """
    Applies Laplacian powers/diffusion in the spectral domain like
        f_out = lambda_i ^ k * e ^ (lambda_i t) f_in
    with learned per-channel parameters k and t.

    Inputs:
      - values: (K,C) in the spectral domain
      - evals: (K) eigenvalues
    Outputs:
      - (K,C) transformed values in the spectral domain
    """

    def __init__(self, C_inout, with_power=True):
        super(LaplacianBlock, self).__init__()
        self.C_inout = C_inout
        self.with_power = with_power

        self.laplacian_power = nn.Parameter(torch.Tensor(C_inout))  # (C)
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)

        if self.with_power:
            nn.init.constant_(self.laplacian_power, 0.0)
        nn.init.constant_(self.diffusion_time, 0.0001)

    def forward(self, x, evals):

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout
                )
            )

        diffusion_coefs = torch.exp(
            -evals.unsqueeze(-1) * torch.abs(self.diffusion_time).unsqueeze(0)
        )

        if self.with_power:
            lambda_coefs = torch.pow(
                evals.unsqueeze(-1),
                (2.0 * torch.sigmoid(self.laplacian_power) - 1.0).unsqueeze(0),
            )
        else:
            lambda_coefs = torch.ones_like(self.laplacian_power)

        if x.is_complex():
            y = ensure_complex(lambda_coefs * diffusion_coefs) * x
        else:
            y = lambda_coefs * diffusion_coefs * x

        return y


def ensure_complex(arr):
    if arr.is_complex():
        return arr
    return arr.to(complex_dtype_equiv(arr.dtype))


def complex_dtype_equiv(d):
    if d == torch.float32:
        return torch.complex64
    elif d == torch.float64:
        return torch.complex128
    else:
        raise RuntimeError("unexpected type: " + str(d))
