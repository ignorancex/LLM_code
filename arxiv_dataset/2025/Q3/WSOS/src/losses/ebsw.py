import torch
from torch import nn

from src.losses.shared import rand_projections


class EBSW(nn.Module):
    def __init__(self, n_sampling: int,p: int = 2,encoder=None ,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.n_sampling = n_sampling
        self.encoder = encoder

    def one_dimensional_Wasserstein_prod(self, X, Y, theta, p):
        X_prod = torch.matmul(X, theta.transpose(0, 1))
        Y_prod = torch.matmul(Y, theta.transpose(0, 1))
        X_prod = X_prod.view(X_prod.shape[0], -1)
        Y_prod = Y_prod.view(Y_prod.shape[0], -1)
        wasserstein_distance = torch.abs(
            (
                    torch.sort(X_prod, dim=0)[0]
                    - torch.sort(Y_prod, dim=0)[0]
            )
        )
        wasserstein_distance = torch.sum(torch.pow(wasserstein_distance, p), dim=0, keepdim=True)
        return wasserstein_distance

    def ISEBSW(self, X, Y):
        dim = X.size(1)
        theta = rand_projections(dim, self.n_sampling).to(X)
        wasserstein_distances = self.one_dimensional_Wasserstein_prod(X, Y, theta, p=self.p)
        wasserstein_distances = wasserstein_distances.view(1, self.n_sampling)
        weights = torch.softmax(wasserstein_distances, dim=1)
        sw = torch.sum(weights * wasserstein_distances, dim=1).mean()
        return torch.pow(sw, 1. / self.p)

    def forward(self, first_sample, second_sample):
        if self.encoder:
            with torch.no_grad():
                first_sample = self.encoder(first_sample*255)
                second_sample = self.encoder(second_sample*255)
        _dswd = self.ISEBSW(
            first_sample.view(first_sample.shape[0], -1),
            second_sample.view(second_sample.shape[0], -1)
        )
        return _dswd