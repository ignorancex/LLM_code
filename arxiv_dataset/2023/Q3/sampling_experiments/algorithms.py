import abc

import numpy as np
import torch
import torch_utils
from typing import *

from torch_utils import DefaultDevice


class Distribution:
    def __init__(self, d: int):
        self.d = d

    def get_space_dimension(self) -> int:
        return self.d

    def sample(self, n_samples: int) -> torch.Tensor:
        raise NotImplementedError()

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.unnorm_log_density(x) - self.log_partition()

    def unnorm_log_density(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def log_partition(self) -> float:
        raise NotImplementedError()


class GibbsDistribution(Distribution):
    def __init__(self, d: int, f: Callable[[torch.Tensor], torch.Tensor]):
        # could have a base distribution,
        # but then the unnorm_log_density would have to be relative to the base distribution
        super().__init__(d)
        self.f = f

    def unnorm_log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)

    def shift_by(self, shift: float) -> 'GibbsDistribution':
        return ShiftedGibbsDistribution(self, shift)


class ShiftedGibbsDistribution(GibbsDistribution):
    def __init__(self, dist: GibbsDistribution, shift: float):
        super().__init__(d=dist.get_space_dimension(), f=lambda x: dist.unnorm_log_density(x)+shift)
        self.dist = dist
        self.shift = shift

    def log_partition(self) -> float:
        return self.dist.log_partition() + self.shift

    def sample(self, n_samples: int) -> torch.Tensor:
        return self.dist.sample(n_samples)

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.dist.log_density(x)


class UniformCubeDistribution(GibbsDistribution):
    # uniform distribution on the unit cube [0, 1]^d
    def __init__(self, d: int):
        super().__init__(d=d, f=lambda x: torch.zeros_like(x[:, 0]))

    def sample(self, n_samples: int):
        return torch.rand(n_samples, self.d, dtype=torch.float64, device=DefaultDevice.get())

    def log_partition(self) -> float:
        return 0.0


def linear_log_partition_1d(param):
    if param < 0:
        return linear_log_partition_1d(-param) + param
    elif np.abs(param) <= 1e-10:
        return 0.0
    elif np.abs(param) <= 10:
        return np.log(np.expm1(param) / param)
    else:
        return np.log1p(-np.exp(-param)) + param - np.log(param)

class LinearGibbsDistribution1D(GibbsDistribution):
    def __init__(self, param: float):
        super().__init__(d=1, f=lambda x: param*x)
        self.param = param
        self.log_partition_value = linear_log_partition_1d(param)

    def log_partition(self):
        return self.log_partition_value

    def sample(self, n_samples: int) -> torch.Tensor:
        unif_samples = torch.rand(n_samples, 1, dtype=torch.float64, device=DefaultDevice.get())
        if np.abs(self.param) <= 1e-10:
            return unif_samples
        else:
            # use the inverse CDF method: indefinite integral is exp(param x) / param
            # normalization constant is (exp(param) - 1)/param
            # hence, CDF is p = (exp(param x) - 1) / (exp(param) - 1)
            # solve for x: x = log(1 + p*(exp(param) - 1)) / param
            # todo: will be unstable for large np.abs(param), but this is not problematic in our experiments
            return torch.log1p(unif_samples * np.expm1(self.param)) / self.param


class TensorProductGibbsDistribution(GibbsDistribution):
    def __init__(self, distributions_1d: List[GibbsDistribution]):
        # could do it in a more general fashion, it is not necessary to assume that the distributions are 1D
        super().__init__(d=len(distributions_1d), f=
            lambda x: sum([distributions_1d[i].unnorm_log_density(x[..., i]) for i in range(len(distributions_1d))]))
        self.distributions_1d = distributions_1d
        self.log_partition_value = sum([dst.log_partition() for dst in distributions_1d])

    def sample(self, n_samples: int) -> torch.Tensor:
        return torch.cat([dst.sample(n_samples) for dst in self.distributions_1d], dim=-1)

    def log_partition(self):
        return self.log_partition_value


class LinearGibbsDistribution(TensorProductGibbsDistribution):
    def __init__(self, params: torch.Tensor):
        super().__init__([LinearGibbsDistribution1D(params[i].item()) for i in range(len(params))])
        self.params = params


class Sampler(abc.ABC):
    def sample(self, gibbs_dist: GibbsDistribution, n_samples: int, n: int) -> torch.Tensor:
        raise NotImplementedError()


class MCSampler(Sampler):
    def __init__(self, proposal_dist: GibbsDistribution):
        self.proposal_dist = proposal_dist

    def sample(self, gibbs_dist: GibbsDistribution, n_samples: int, n: int) -> torch.Tensor:
        # use batching to avoid RAM overflow
        max_per_batch = 100000000  # max 0.8 GB per Tensor
        max_n_samples_per_batch = min(n_samples, max(1, max_per_batch // n))
        all_samples = []
        n_all_samples = 0
        while n_all_samples < n_samples:
            n_batch_samples = min(max_n_samples_per_batch, n_samples - n_all_samples)

            proposal_samples = self.proposal_dist.sample(n_batch_samples * n)
            log_diffs = gibbs_dist.unnorm_log_density(proposal_samples) \
                        - self.proposal_dist.unnorm_log_density(proposal_samples)
            log_diffs = log_diffs.view(n_batch_samples, n)
            proposal_samples = proposal_samples.view(n_batch_samples, n, -1)
            log_diffs -= torch.max(log_diffs, dim=-1, keepdim=True).values
            exp_log_diffs = torch.exp(log_diffs)
            idxs = torch.multinomial(exp_log_diffs, 1, replacement=True).squeeze(-1)
            all_samples.append(proposal_samples[torch.arange(n_batch_samples), idxs, :])
            n_all_samples += n_batch_samples

        return torch.cat(all_samples, dim=0)


class RejectionSampler(Sampler):
    def __init__(self, proposal_dist: Distribution):
        self.proposal_dist = proposal_dist
        pass

    def sample(self, gibbs_dist: GibbsDistribution, n_samples: int, n: int) -> torch.Tensor:
        unterminated = torch.arange(n_samples, dtype=torch.long, device=DefaultDevice.get())
        n_unterminated = n_samples
        step = 0
        samples = torch.zeros(n_samples, gibbs_dist.get_space_dimension(), dtype=torch.float64,
                              device=DefaultDevice.get())
        while (n < 0 or step < n) and n_unterminated > 0:
            new_samples = self.proposal_dist.sample(n_unterminated)
            unif_weights = torch.rand(n_unterminated, dtype=torch.float64, device=DefaultDevice.get())
            log_diff = gibbs_dist.unnorm_log_density(new_samples) \
                       - self.proposal_dist.unnorm_log_density(new_samples)
            accept = unif_weights < torch.exp(log_diff)
            n_unterminated -= torch.count_nonzero(accept).item()
            samples[unterminated[accept]] = new_samples[accept]
            unterminated = unterminated[~accept]
            step += 1
        # print(f'Number of rejection sampling steps: {step:g}')

        if n_unterminated > 0:
            samples[unterminated] = self.proposal_dist.sample(n_unterminated)

        return samples


class LogPartitionEstimator:
    def eval(self, gibbs_dist: GibbsDistribution, n_batch: int, n: int) -> torch.Tensor:
        raise NotImplementedError()


class MCLogPartition(LogPartitionEstimator):
    def __init__(self, proposal_dist: GibbsDistribution):
        self.proposal_dist = proposal_dist

    def eval(self, gibbs_dist: GibbsDistribution, n_reps: int, n: int) -> torch.Tensor:
        # use batching to avoid RAM overflow
        max_per_batch = 100000000  # max 0.8 GB per Tensor
        max_n_reps_per_batch = min(n_reps, max(1, max_per_batch // n))
        all_reps = []
        n_all_reps = 0
        while n_all_reps < n_reps:
            n_batch_reps = min(max_n_reps_per_batch, n_reps - n_all_reps)
            proposal_samples = self.proposal_dist.sample(n_batch_reps * n)
            log_diff = gibbs_dist.unnorm_log_density(proposal_samples) \
                        - self.proposal_dist.unnorm_log_density(proposal_samples)
            all_reps.append(self.proposal_dist.log_partition()
                            + torch_utils.logmeanexp(log_diff.view(n_batch_reps, n), dim=1))
            n_all_reps += n_batch_reps
        return torch.cat(all_reps, dim=0)


def mc_logpartition_upper_bound(n: int, d: int, lip_const: float, delta: float):
    """
    Upper bound for the error of MC logpartition from the paper.
    :param n: Number of samples used to compute the estimate.
    :param d: Dimension of the cube.
    :param lip_const: Lipschitz constant of the target function.
    :param delta: provide the bound that holds with probability at least 1-delta
    :return:
    """
    threshold = 4*np.log(2/delta)*(1+3*d**(-1/2)*lip_const)**d
    if n < threshold:
        first_term = d**(1/2) * (np.log(1/delta) ** (1/d)) * lip_const * n ** (-1/d)
        second_term = np.log(4 * np.log(2 / delta))
        third_term = d * np.log(1 + 3 * d**(-1/2) * lip_const)
        return first_term + second_term + third_term
    else:
        return 4 * np.log(2/delta) ** (1/2) * (1 + 3 * d**(-1/2) * lip_const) ** (d/2) * n**(-1/2)


class PiecewiseConstantGibbsDistribution(GibbsDistribution):
    def __init__(self, to_approximate: GibbsDistribution, n_per_dim: int):
        # f is just a dummy and will be overridden
        super().__init__(d=to_approximate.get_space_dimension(), f=lambda x: x)
        self.n_per_dim = n_per_dim
        self.n = n_per_dim ** self.d
        self.grid_1d = (0.5 + torch.arange(n_per_dim, dtype=torch.float64, device=DefaultDevice.get())) / n_per_dim
        # self.grid = torch_utils.tensor_prod(*([self.grid_1d] * self.d))
        # self.grid = torch.zeros(*([n_per_dim] * self.d + [self.d]))
        # self.grid is a [n_per_dim] * d + [d] grid
        self.grid = torch.stack(list(torch.meshgrid(*([self.grid_1d] * self.d), indexing='ij')), dim=-1)
        # shape: self.n x self.d
        self.flattened_grid = self.grid.view(-1, self.d)
        # shape: self.n
        self.flattened_function_values = to_approximate.unnorm_log_density(self.flattened_grid)
        self.function_values = self.flattened_function_values.view(*([self.n_per_dim]*self.d))
        self.log_partition_value = torch_utils.logmeanexp(self.flattened_function_values, dim=0).item()
        self.flattened_probabilities = torch.exp(self.flattened_function_values - self.log_partition_value)

    def log_partition(self) -> float:
        return self.log_partition_value

    def sample(self, n_samples: int) -> torch.Tensor:
        cell_samples = torch.rand(n_samples, self.d, dtype=torch.float64, device=DefaultDevice.get()) / self.n_per_dim
        flattened_idxs = torch.multinomial(self.flattened_probabilities, num_samples=n_samples, replacement=True)
        rev_dim_idxs = []
        for i in range(self.d):
            rev_dim_idxs.append(flattened_idxs % self.n_per_dim)
            flattened_idxs = torch.div(flattened_idxs, self.n_per_dim, rounding_mode='floor')
        expanded_idxs = torch.stack(list(reversed(rev_dim_idxs)), dim=-1)
        cell_origins = expanded_idxs / self.n_per_dim
        return cell_origins + cell_samples

    def unnorm_log_density(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=0.0, max=1.0 - 1e-8)
        idxs = torch.clamp((self.n_per_dim * x).long(), min=0, max=self.n_per_dim-1)
        return self.function_values[[idxs[:, i] for i in range(self.d)]]

