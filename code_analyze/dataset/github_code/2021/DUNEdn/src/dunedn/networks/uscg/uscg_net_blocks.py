"""
    This module contains the USCG Net building blocks.
"""
import torch
from torch import nn


class SCG_Block(nn.Module):
    def __init__(
        self, in_ch, hidden_ch=6, node_size=(32, 32), add_diag=True, dropout=0.2
    ):
        super(SCG_Block, self).__init__()
        self.node_size = node_size
        self.hidden = hidden_ch
        self.nodes = node_size[0] * node_size[1]
        self.add_diag = add_diag
        self.pool = nn.AdaptiveMaxPool2d(node_size)

        self.mu = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=True),
            nn.Dropout(dropout),
        )

        self.logvar = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 1, 1, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, gx):
        B, C, H, W = gx.size()
        mu, log_var = self.mu(gx), self.logvar(gx)
        if self.training:
            std = torch.exp(log_var.reshape(B, self.nodes, self.hidden))
            eps = torch.randn_like(std)
            z = mu.reshape(B, self.nodes, self.hidden) + std * eps
        else:
            z = mu.reshape(B, self.nodes, self.hidden)
        A = torch.matmul(z, z.permute(0, 2, 1))
        A = torch.relu(A)
        Ad = torch.diagonal(A, dim1=1, dim2=2)
        mean = torch.mean(Ad, dim=1)
        gamma = torch.sqrt(1 + 1.0 / mean).unsqueeze(-1).unsqueeze(-1)
        if self.training:
            dl_loss = (
                gamma.mean()
                * torch.log(Ad[Ad < 1] + 1.0e-7).sum()
                / (A.size(0) * A.size(1) * A.size(2))
            )
            kl_loss = (
                -0.5
                / self.nodes
                * torch.mean(
                    torch.sum(1 + 2 * log_var - mu.pow(2) - log_var.exp().pow(2), 1)
                )
            )
        loss = kl_loss - dl_loss if self.training else None
        if self.add_diag:
            diag = []
            for i in range(Ad.shape[0]):
                diag.append(torch.diag(Ad[i, :]).unsqueeze(0))
            A = A + gamma * torch.cat(diag, 0)
        A = self.laplacian_matrix(A, self_loop=True)
        z_hat = (
            gamma.mean()
            * mu.reshape(B, self.nodes, self.hidden)
            * (1.0 - log_var.reshape(B, self.nodes, self.hidden))
        )
        return A, gx, loss, z_hat

    def laplacian_matrix(self, A, self_loop=False):
        """
        Computes normalized Laplacian matrix: A (B, N, N)
        """
        if self_loop:
            A = A + torch.eye(A.size(1), device=A.device).unsqueeze(0)
        deg_inv_sqrt = (torch.sum(A, 1) + 1e-5).pow(-0.5)
        LA = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
        return LA


class GCN_Layer(nn.Module):
    def __init__(
        self, in_features, out_features, bnorm=True, activation=nn.ReLU(), dropout=None
    ):
        super(GCN_Layer, self).__init__()
        self.bnorm = bnorm
        fc = [nn.Linear(in_features, out_features)]
        if bnorm:
            fc.append(BatchNorm_GCN(out_features))
        if activation is not None:
            fc.append(activation)
        if dropout is not None:
            fc.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        x, A = data
        y = self.fc(torch.bmm(A, x))

        return [y, A]


class Pooling_Block(nn.Module):
    def __init__(self, c, h, w):
        """
        Parameters:
            c: int, image channels
            h: int, output height
            w: int, output width
        """
        super(Pooling_Block, self).__init__()
        self.pooling = nn.Sequential(
            nn.AdaptiveMaxPool2d((h, w)),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.pooling(x)


class Recombination_Layer(nn.Module):
    """
    This layer recombines an output with a residual connection through a (1,1)
    convolution. It first concatenates the two inputs along the channel
    dimension and then it applies the convolution.
    """

    def __init__(self):
        super(Recombination_Layer, self).__init__()
        self.conv = nn.Conv2d(2, 1, 1)

    def forward(self, x, y):
        return self.conv(torch.cat([x, y], axis=1))


# ==============================================================================
# functions and classes to be called within this module only


class BatchNorm_GCN(nn.BatchNorm1d):
    """Batch normalization over GCN features"""

    def __init__(self, num_features):
        super(BatchNorm_GCN, self).__init__(num_features)

    def forward(self, x):
        return super(BatchNorm_GCN, self).forward(x.permute(0, 2, 1)).permute(0, 2, 1)
