import time
from typing import List, Tuple, Optional

# import geomloss
import torch
import numpy as np


def get_batch_intervals(n_total: int, batch_size: int) -> List[Tuple[int, int]]:
    boundaries = [i * batch_size for i in range(1 + n_total // batch_size)]
    if boundaries[-1] != n_total:
        boundaries.append(n_total)
    return [(start, stop) for start, stop in zip(boundaries[:-1], boundaries[1:])]


def expected_dist(samples_1: torch.Tensor, samples_2: torch.Tensor, max_matrix_elements: int = 100000000) -> float:
    torch.backends.cuda.matmul.allow_tf32 = False
    diag_1 = samples_1.square().sum(dim=-1)
    diag_2 = samples_2.square().sum(dim=-1)
    batch_size_2 = max(1, max_matrix_elements // samples_1.shape[0])
    dist_sum = 0.0
    for start, stop in get_batch_intervals(samples_2.shape[0], batch_size_2):
        # print(samples_1, samples_2)
        matrix_block = samples_1 @ samples_2[start:stop, :].t()
        sq_dists = diag_1[:, None] + diag_2[None, start:stop] - 2 * matrix_block
        dist_sum = dist_sum + sq_dists.clamp(min=0.).sqrt().sum()
    return dist_sum.item() / (samples_1.shape[0] * samples_2.shape[0])


def expected_dist_symm(samples_1: torch.Tensor, max_matrix_elements: int = 100000000) -> float:
    samples_2 = samples_1
    torch.backends.cuda.matmul.allow_tf32 = False
    diag_1 = samples_1.square().sum(dim=-1)
    diag_2 = diag_1
    batch_size_2 = max(1, max_matrix_elements // samples_1.shape[0])
    dist_sum = 0.0
    for start, stop in get_batch_intervals(samples_2.shape[0], batch_size_2):
        matrix_block = samples_1[start:] @ samples_2[start:stop, :].t()
        sq_dists = diag_1[start:, None] + diag_2[None, start:stop] - 2 * matrix_block
        dists = sq_dists.clamp(min=0.).sqrt()
        # add 2 * the block-off-diagonal part since it is not counted twice
        dist_sum = dist_sum + dists[:(stop-start), :].sum() + 2*dists[(stop-start):, :].sum()
    return dist_sum.item() / (samples_1.shape[0] * samples_2.shape[0])


def expected_dist_2(samples_1: torch.Tensor, samples_2: torch.Tensor, max_matrix_elements: int = 100000000) -> float:
    batch_size_2 = max(1, max_matrix_elements // samples_1.shape[0])
    dist_sum = 0.0
    for start, stop in get_batch_intervals(samples_2.shape[0], batch_size_2):
        dist_sum = dist_sum + (samples_1[:, None, :] - samples_2[None, start:stop, :]).norm(dim=-1).sum()
    return dist_sum.item() / (samples_1.shape[0] * samples_2.shape[0])


def expected_dist_2_symm(samples_1: torch.Tensor, max_matrix_elements: int = 100000000) -> float:
    samples_2 = samples_1
    batch_size_2 = max(1, max_matrix_elements // samples_1.shape[0])
    dist_sum = 0.0
    for start, stop in get_batch_intervals(samples_2.shape[0], batch_size_2):
        dists = (samples_1[start:, None, :] - samples_2[None, start:stop, :]).norm(dim=-1)
        dist_sum = dist_sum + dists[:(stop-start), :].sum() + 2*dists[(stop-start):, :].sum()
    return dist_sum.item() / (samples_1.shape[0] * samples_2.shape[0])


def energy_dist(samples_1: torch.Tensor, samples_2: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
    if device is not None:
        samples_1 = samples_1.to(device)
        samples_2 = samples_2.to(device)
    # use higher precision, which is important for sizes >= 100k samples
    samples_1 = samples_1.double()
    samples_2 = samples_2.double()
    sq_energy_dist = 2*expected_dist(samples_1, samples_2) \
                     - expected_dist_symm(samples_1) - expected_dist_symm(samples_2)
    # sq_energy_dist = 2*expected_dist(samples_1, samples_2) \
    #                  - expected_dist(samples_1, samples_1) - expected_dist(samples_2, samples_2)
    # sq_energy_dist = 2 * expected_dist_2(samples_1, samples_2) \
    #                  - expected_dist_2(samples_1, samples_1) - expected_dist_2(samples_2, samples_2)
    # sq_energy_dist = 2 * expected_dist_2(samples_1, samples_2) \
    #                  - expected_dist_2_symm(samples_1) - expected_dist_2_symm(samples_2)
    return torch.as_tensor(np.sqrt(max(0.0, sq_energy_dist)))


if __name__ == '__main__':
    n_samples = 10000
    device = 'cpu'
    torch.manual_seed(1234)
    samples_1 = torch.randn(n_samples, 1, dtype=torch.float32).to(device)
    samples_2 = torch.randn(n_samples, 1, dtype=torch.float32).to(device)
    #samples_2 = samples_1
    loss = energy_dist
    # loss = geomloss.SamplesLoss(loss='sinkhorn', p=1, blur=1e-3, backend='multiscale')
    start_time = time.time()
    loss_value = loss(samples_1, samples_2)
    end_time = time.time()
    print(f'Obtained loss value: {loss_value:g}')
    print(f'Time: {end_time-start_time:g} s')
