# 3p
import torch.nn as nn
# project
from .layers import TSNBlock_Scalar


class DiffusionNet(nn.Module):
    def __init__(self, dim_in=3, dim_out=128, n_width=128, n_block=4, pairwise_dot=True, dot_linear_complex=True, dropout=False, neig=128, **kwargs):
        super().__init__()

        self.neig = neig
        self.blocks = []
        hidden_dims = [n_width, n_width]
        for i_block in range(n_block):
            self.blocks.append(
                TSNBlock_Scalar(C0_inout=n_width,
                                C0_hidden=hidden_dims,
                                pairwise_dot=pairwise_dot,
                                dot_linear_complex=dot_linear_complex,
                                dropout=dropout,
                                **kwargs
                                )
            )
            self.add_module("block_" + str(i_block), self.blocks[-1])

        self.first_lin = nn.Linear(dim_in, n_width)
        self.last_lin = nn.Linear(n_width, dim_out)

    def forward(self, x, mass, evals, evecs, grad_from_spectral):
        x, evals, evecs = x.float(), evals[:, :self.neig].float(), evecs[:, :, :self.neig].float()
        mass, grad_from_spectral = mass.float(), grad_from_spectral[:, :, :self.neig, :].float()

        x = self.first_lin(x)

        for block in self.blocks:
            x = block(x, mass, evals, evecs, grad_from_spectral)

        # Apply the last linear layer
        x = self.last_lin(x)

        # output data
        return x
