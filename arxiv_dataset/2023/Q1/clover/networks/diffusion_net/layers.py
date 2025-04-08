# 3p
import torch
import torch.nn as nn
# project
from .utils import to_basis, from_basis, ensure_complex, cmatvecmul_stacked


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

    def __init__(self, C_inout, with_power=True, max_time=False):
        super(LaplacianBlock, self).__init__()
        self.C_inout = C_inout
        self.with_power = with_power
        self.max_time = max_time

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

        if self.max_time:
            diffusion_time = self.max_time * torch.sigmoid(self.diffusion_time)
            # diffusion_time = self.diffusion_time.clamp(min=-self.max_time, max=self.max_time)
        else:
            diffusion_time = self.diffusion_time

        diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * torch.abs(diffusion_time).unsqueeze(0))

        if self.with_power:
            lambda_coefs = torch.pow(evals.unsqueeze(-1), (2.0 * torch.sigmoid(self.laplacian_power) - 1.0).unsqueeze(0))
        else:
            lambda_coefs = torch.ones_like(self.laplacian_power)

        if x.is_complex():
            y = ensure_complex(lambda_coefs * diffusion_coefs) * x
        else:
            y = lambda_coefs * diffusion_coefs * x

        return y


class PairwiseDot(nn.Module):
    """
    Compute dot-products between input vectors. Uses a learned complex-linear layer to keep dimension down.

    Input:
        - vectors: (V,C,2)
    Output:
        - dots: (V,C) dots
    """

    def __init__(self, C_inout, linear_complex=True):
        super(PairwiseDot, self).__init__()

        self.C_inout = C_inout
        self.linear_complex = linear_complex

        if self.linear_complex:
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)

    def forward(self, vectors):

        vectorsA = vectors  # (V,C)

        if self.linear_complex:
            vectorsBreal = self.A_re(vectors[..., 0]) - self.A_im(vectors[..., 1])
            vectorsBimag = self.A_re(vectors[..., 1]) + self.A_im(vectors[..., 0])
        else:
            vectorsBreal = self.A(vectors[..., 0])
            vectorsBimag = self.A(vectors[..., 1])

        dots = vectorsA[..., 0] * vectorsBreal + vectorsA[..., 1] * vectorsBimag

        return torch.tanh(dots)


class MiniMLP(nn.Sequential):
    def __init__(
        self,
        layer_sizes,
        dropout=False,
        activation=nn.ReLU,
        batch_norm=False,
        name="miniMLP",
    ):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = i + 2 == len(layer_sizes)

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i), nn.Dropout(p=0.5)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(layer_sizes[i], layer_sizes[i + 1],),
            )

            if batch_norm and not is_last:
                self.add_module(
                    name + "_mlp_batch_norm_{:03d}".format(i),
                    BatchNormLastDim(layer_sizes[i + 1]),
                )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(name + "_mlp_act_{:03d}".format(i), activation())


class BatchNormLastDim(nn.Module):
    def __init__(self, s):
        super(BatchNormLastDim, self).__init__()
        self.s = s
        self.bn = nn.BatchNorm1d(s)

    def forward(self, x):
        init_dim = x.shape
        if init_dim[-1] != self.s:
            raise ValueError(
                "batch norm last dim does not have right shape. should be {}, but is {}".format(
                    self.s, init_dim[-1]
                )
            )

        x_flat = x.view((-1, self.s))
        bn_flat = self.bn(x_flat)
        return bn_flat.view(*init_dim)


class TSNBlock_Scalar(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(
        self,
        C0_inout,
        C0_hidden,
        dropout=False,
        pairwise_dot=True,
        with_power=False,
        max_time=False,
        dot_linear_complex=True,
        grad_method="spectral_pointwise",
    ):
        super(TSNBlock_Scalar, self).__init__()

        # Specified dimensions
        self.C0_inout = C0_inout
        self.C0_hidden = C0_hidden

        self.dropout = dropout
        self.pairwise_dot = pairwise_dot
        self.with_power = with_power
        self.max_time = max_time
        self.dot_linear_complex = dot_linear_complex
        self.grad_method = grad_method  # one of ['pointwise', 'spectral_pointwise', 'spectral_spectral']

        # Laplacian block
        self.spec0 = LaplacianBlock(self.C0_inout, self.with_power, self.max_time)

        self.C0_mlp = 2 * self.C0_inout

        if self.pairwise_dot:
            if self.grad_method == "pointwise":
                self.pairwise_dot = PairwiseDot(
                    self.C0_inout, linear_complex=self.dot_linear_complex
                )
                self.C0_mlp += self.C0_inout
            elif self.grad_method == "spectral_pointwise":
                self.pairwise_dot = PairwiseDot(
                    self.C0_inout, linear_complex=self.dot_linear_complex
                )
                self.C0_mlp += self.C0_inout
            elif self.grad_method == "spectral_spectral":
                self.pairwise_dot = PairwiseDot(
                    self.C0_inout, linear_complex=self.dot_linear_complex
                )
                self.C0_mlp += self.C0_inout

        # MLPs
        self.mlp0 = MiniMLP([self.C0_mlp] + self.C0_hidden + [self.C0_inout], dropout=self.dropout)

    def forward(self, x0, mass, evals, evecs, grad_from_spectral):

        if x0.shape[-1] != self.C0_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x0.shape, self.C0_inout
                )
            )

        # Transform to spectral
        x0_spec = to_basis(x0, evecs, mass)  # (K, C0_in)r

        # Laplacian block
        x0_spec = self.spec0(x0_spec, evals)

        # Transform back to per-vertex
        x0_lap = from_basis(x0_spec, evecs)
        x0_comb = torch.cat(
            (x0, x0_lap), dim=-1
        )  # (V, C0_speccomb + C0_in)r = (V, C0_mlp)r

        if self.pairwise_dot:
            # If using the pairwise dot block, add it to the scalar values as well

            if self.grad_method == "spectral_pointwise":
                x0_grad = cmatvecmul_stacked(grad_from_spectral, x0_spec)
            elif self.grad_method == "pointwise":
                pass
            elif self.grad_method == "spectral_spectral":
                pass

            x0_gradprods = self.pairwise_dot(x0_grad)
            x0_comb = torch.cat((x0_comb, x0_gradprods), dim=-1)

        # Apply the mlp
        x0_out = self.mlp0(x0_comb)

        # Skip connection
        x0_out = x0_out + x0

        return x0_out
