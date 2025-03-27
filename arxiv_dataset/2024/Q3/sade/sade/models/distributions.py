import torch
import torch.nn as nn
from torch.distributions import (
    LowRankMultivariateNormal,
    MixtureSameFamily,
    MultivariateNormal,
)


class MVN(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.D = D = n_features
        lower_tril_numel = D * (D + 1) // 2 - D
        self.cov_factor = nn.Parameter(torch.randn(lower_tril_numel), requires_grad=True)
        self.cov_diag = nn.Parameter(torch.ones(D), requires_grad=True)
        self.mean = nn.Parameter(torch.zeros(D), requires_grad=True)

        self.tril_idx = torch.tril_indices(D, D, offset=-1).tolist()

    @property
    def diag(self):
        return torch.nn.functional.softplus(self.cov_diag + 1e-5).diag()

    @property
    def scale_tril(self):
        tril = torch.zeros(
            self.D, self.D, requires_grad=False, device=self.cov_factor.device
        )
        tril[self.tril_idx] = self.cov_factor
        return self.diag + tril

    def forward(self, x):
        return MultivariateNormal(loc=self.mean, scale_tril=self.scale_tril).log_prob(x)

    def log_prob(self, x):
        return self(x)


class GMM(nn.Module):
    def __init__(self, n_components, n_features):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features

        # Initialize the parameters
        self.weights = nn.Parameter(torch.ones(n_components))
        self.means = nn.Parameter(torch.zeros(n_components, self.n_features))

        lower_tril_numel = self.n_features * (self.n_features + 1) // 2
        self.scale_tril_elements = nn.Parameter(
            torch.ones(n_components, lower_tril_numel), requires_grad=True
        )
        self.tril_idx = torch.tril_indices(self.n_features, self.n_features).tolist()

    # Ensure scale_tril is lower triangular
    @property
    def scale_tril(self):
        # Convert scale_tril_elements to a lower triangular matrix
        scale_tril = torch.zeros(
            self.n_components,
            self.n_features,
            self.n_features,
            device=self.scale_tril_elements.device,
        )
        scale_tril[:, self.tril_idx[0], self.tril_idx[1]] = self.scale_tril_elements

        # # Ensure the diagonal of scale_tril is positive
        scale_tril = scale_tril.tril(-1) + torch.diag_embed(
            torch.nn.functional.softplus(scale_tril.diagonal(dim1=-2, dim2=-1))
        )
        return scale_tril

    def forward(self, x):
        # Create the component distributions
        component_dists = MultivariateNormal(self.means, scale_tril=self.scale_tril)

        # Create the mixture distribution
        mixture_dist = MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(logits=self.weights),
            component_distribution=component_dists,
        )

        return mixture_dist.log_prob(x)

    def log_prob(self, x):
        return self(x)


class LowRankMVN(nn.Module):
    def __init__(self, n_features, low_rank_dims=256):
        super().__init__()

        self.cov_factor = nn.Parameter(
            torch.randn(n_features, low_rank_dims), requires_grad=True
        )
        self.cov_diag = nn.Parameter(torch.ones(n_features), requires_grad=True)
        self.mean = nn.Parameter(torch.zeros(n_features), requires_grad=False)

    @property
    def diag(self):
        return torch.nn.functional.softplus(self.cov_diag) + 1e-5

    @property
    def covariance_matrix(self):
        return LowRankMultivariateNormal(
            loc=self.mean, cov_factor=self.cov_factor, cov_diag=self.diag
        ).covariance_matrix

    def forward(self, x):
        return LowRankMultivariateNormal(
            loc=self.mean, cov_factor=self.cov_factor, cov_diag=self.diag
        ).log_prob(x)

    def log_prob(self, x):
        return self(x)
