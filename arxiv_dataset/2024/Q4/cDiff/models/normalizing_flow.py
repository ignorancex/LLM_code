import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


class PermutationLayer(nn.Module):
    def __init__(self, size):
        super(PermutationLayer, self).__init__()
        self.permutation_vec = nn.Parameter(torch.randperm(size), requires_grad=False)
        _, inverse_permutation = torch.sort(self.permutation_vec)
        self.inv_permutation_vec = nn.Parameter(inverse_permutation, requires_grad=False)

    def forward(self, x, inverse=False):
        if not inverse:
            return x[..., self.permutation_vec]
        else:
            return x[..., self.inv_permutation_vec]


class Conditioner(nn.Module):
    def __init__(
            self, in_dim: int, out_dim: int,
            n_layers: int, hidden_dim: int,
            n_params=2
    ):
        super(Conditioner, self).__init__()
        self.n_params = n_params
        self.out_dim = out_dim
        # the input, interior, and output layers
        self.input = nn.Linear(in_dim, hidden_dim)
        self.hidden = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        # output layer needs out_dim*n_params outputs
        # (e.g. n_params=2 for a scale and translation output)
        self.output = nn.Linear(hidden_dim, out_dim * n_params)

    def forward(self, x):
        x = F.elu(self.input(x))
        for h in self.hidden:
            x = x + F.elu(h(x))
        x = self.output(x)
        # returns a list of conditioning parameters
        # there will be n_params items in the list, one per parameter
        # and therefore the size of each chunk is self.out_dim
        return torch.split(x, split_size_or_sections=self.out_dim, dim=1)


class ConditionalAffineCouplingLayer(nn.Module):
    # Notation for x, y, z:
    # x: sample in original space given y, from p(x | y)
    # y: conditioning input (i.e. the data from p(y | x)
    # z: sample in normalized space, from p(z | y), z =d f(x, y)
    def __init__(self,
                 n_in,  # input size into the affine coupling layer, also n_out
                 n_y,  # size of the conditioning input
                 n_cond_layers, hidden_dim,
                 n_params=2,
                 permute=True,
                 alpha=None  # soft clamping of the scales in the affine transformations
                 ):
        super(ConditionalAffineCouplingLayer, self).__init__()
        self.n_in = n_in
        self.permute = permute
        self.alpha = alpha

        # create the conditioning functions
        # each conditioner sees a block of the input, plus x (conditioning input)
        # input u = (u1, u2) in R^dim1 X R^dim2
        # output v = (v1, v2) partitioned as u
        dim2 = (n_in // 2)
        dim1 = (n_in - dim2)

        self.theta1 = Conditioner(in_dim=dim2 + n_y,
                                  out_dim=dim1,
                                  n_layers=n_cond_layers,
                                  hidden_dim=hidden_dim,
                                  n_params=n_params)
        self.theta2 = Conditioner(in_dim=dim1 + n_y,
                                  out_dim=dim2,
                                  n_layers=n_cond_layers,
                                  hidden_dim=hidden_dim,
                                  n_params=n_params)

        if self.permute:
            self.permutation = PermutationLayer(size=self.n_in)

    def f(self, u, y):
        '''in the f : (x, y) -> z direction. The normalizing flow.'''
        if u.size(1) < 2:
            u1 = u
            u2 = u
        else:
            u1, u2 = torch.chunk(u, chunks=2, dim=1)
        t1, s1 = self.theta1(torch.cat((u2, y), dim=1))
        if self.alpha is not None:
            s1 = (2.0 * self.alpha / np.pi) * torch.atan(s1 / self.alpha)
        v1 = u1 * torch.exp(s1) + t1
        t2, s2 = self.theta2(torch.cat((v1, y), dim=1))
        if self.alpha is not None:
            s2 = (2.0 * self.alpha / np.pi) * torch.atan(s2 / self.alpha)
        v2 = u2 * torch.exp(s2) + t2
        log_det = s1.sum(-1) + s2.sum(-1)
        v = torch.cat((v1, v2), dim=-1)
        if self.permute:
            v = self.permutation(v)
        return v, log_det

    def g(self, v, y):
        '''in the g : (z, y) -> x direction. The inverse of the normalizing flow.'''
        if self.permute:
            v = self.permutation(v, inverse=True)
        v1, v2 = torch.chunk(v, chunks=2, dim=1)
        t2, s2 = self.theta2(torch.cat((v1, y), dim=1))
        if self.alpha is not None:
            s2 = (2.0 * self.alpha / np.pi) * torch.atan(s2 / self.alpha)
        u2 = (v2 - t2) * torch.exp(-s2)
        t1, s1 = self.theta1(torch.cat((u2, y), dim=1))
        if self.alpha is not None:
            s1 = (2.0 * self.alpha / np.pi) * torch.atan(s1 / self.alpha)
        u1 = (v1 - t1) * torch.exp(-s1)
        u = torch.cat((u1, u2), dim=-1)
        return u


class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, x_dim, y_dim, n_cond_layers, hidden_dim, n_params, n_flows, alpha=1.0):
        super().__init__()

        # register a dummy buffer to allow propagation of device changes
        self.register_buffer("dummy", torch.tensor(0.0))

        # distribution for the latent variable
        self.register_buffer("mean", torch.zeros(x_dim))
        self.register_buffer("cov", torch.eye(x_dim))
        self.latent = MultivariateNormal(self.mean, self.cov)

        # Create the flows
        flows = nn.ModuleList()
        for _ in range(n_flows):
            flows.append(ConditionalAffineCouplingLayer(n_in=x_dim, n_y=y_dim,
                                                        n_cond_layers=n_cond_layers, hidden_dim=hidden_dim,
                                                        n_params=n_params,
                                                        permute=True, alpha=alpha))
        self.flows = flows

    def latent_log_prob(self, z: torch.Tensor):
        return self.latent.log_prob(z)

    def latent_sample(self, num_samples: int = 1):
        return self.latent.sample((num_samples,))

    @torch.no_grad()
    def sample(self, y: torch.Tensor):
        # Sample a new observation from p(x | y) by sampling z from
        # the latent distribution and passing (z, y) through g.
        num_samples = y.size(0)
        z = self.latent_sample(num_samples)
        z = z.to(self.dummy.device, self.dummy.dtype)
        return self.inverse(z, y)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        '''Maps (x, y) to latent variable z.
        Additionally, computes the log determinant
        of the Jacobian for this transformation.
        '''
        sum_log_abs_det = torch.zeros(x.size(0)).to(self.dummy.device)
        for flow in self.flows:
            x, log_abs_det = flow.f(x, y)
            sum_log_abs_det += log_abs_det

        return x, sum_log_abs_det

    def inverse(self, z: torch.Tensor, y: torch.Tensor):
        '''Maps (z, y) to observation x.
        '''
        with torch.no_grad():
            x = z
            x = x.to(self.dummy.device)
            for flow in reversed(self.flows):
                x = flow.g(x, y)

        return x

    def log_prob(self, x: torch.Tensor, y: torch.Tensor):
        '''Computes log p(x | y) using the change of variable formula.'''
        z, log_abs_det = self(x, y)
        return self.latent_log_prob(z) + log_abs_det

    def __len__(self):
        return len(self.flows)

    # need a custom .to() method in order to ensure that self.latent is moved to the correct device
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = args[0] if len(args) > 0 else kwargs.get('device', None)
        if device is not None:
            self.latent = MultivariateNormal(self.mean.to(device), self.cov.to(device))
        return self