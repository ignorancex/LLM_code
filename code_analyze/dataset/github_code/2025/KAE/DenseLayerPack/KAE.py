import torch
import torch.nn as nn


class KAELayer(nn.Module):
    def __init__(self, input_dim, out_dim, order, addbias=True):
        super(KAELayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.order = order + 1
        self.addbias = addbias
        self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, order + 1) * 0.01)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x):
        shape = x.shape
        outshape = shape[0:-1] + (self.out_dim,)
        x = torch.reshape(x, (-1, self.input_dim))

        x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1)

        y = torch.zeros((x.shape[0], self.out_dim), device=x.device)

        for i in range(self.order):
            term = (x_expanded**i) * self.coeffs[:, :, i]
            y += term.sum(dim=-1)

        if self.addbias:
            y += self.bias

        y = torch.reshape(y, outshape)
        return y

    def regularization_loss(self):
        pass
