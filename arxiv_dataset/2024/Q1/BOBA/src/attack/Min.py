import torch
from math import ceil
from scipy.stats import norm
from .Byzantine import Byzantine


class MinMax(Byzantine):
    def __init__(self, args):
        super(MinMax, self).__init__(args)

        # compute the normal distribution quantile
        self.gamma_init = args.gamma_init
        self.min_tau = args.min_tau

    def attack(self, model, matrix):
        num_honest, length = matrix.shape

        # calculate mean and standard deviation

        mean = matrix.mean(dim=0)
        std = matrix.std(dim=0)  # torch.std by default computes the group std, which is an unbiased estimator

        # proprocessing
        max_dist = 0
        for i in range(num_honest - 1):
            curr_max = (matrix[i] - matrix[i + 1:]).norm(dim=1).max().item()
            max_dist = max(max_dist, curr_max)

        step = self.gamma_init / 2
        gamma = self.gamma_init
        gamma_succ = 0  # initialize as a must-win
        tau = self.min_tau

        while abs(gamma - gamma_succ) > tau:

            vector = mean - gamma * std
            byz_dist = (vector - matrix).norm(dim=1).max().item()

            if byz_dist <= max_dist:  # succeed to be stealthy
                gamma_succ = gamma
                gamma = gamma + step

            else:  # fail
                gamma = gamma - step

            step /= 2

        vector = mean - gamma_succ * std
        vectors = vector.repeat(self.num_byz, 1)

        return vectors


class MinSum(Byzantine):
    def __init__(self, args):
        super(MinSum, self).__init__(args)

        # compute the normal distribution quantile
        self.gamma_init = args.gamma_init
        self.min_tau = args.min_tau

    def attack(self, model, matrix):
        num_honest, length = matrix.shape

        # calculate mean and standard deviation

        mean = matrix.mean(dim=0)
        std = matrix.std(dim=0)  # torch.std by default computes the group std, which is an unbiased estimator

        # proprocessing
        max_dist = 0
        for i in range(num_honest - 1):
            curr = (matrix[i] - matrix).norm(dim=1).square().sum().item()
            max_dist = max(max_dist, curr)

        step = self.gamma_init / 2
        gamma = self.gamma_init
        gamma_succ = 0  # initialize as a must-win
        tau = self.min_tau

        while abs(gamma - gamma_succ) > tau:

            vector = mean - gamma * std
            byz_dist = (vector - matrix).norm(dim=1).square().sum().item()

            if byz_dist <= max_dist:  # succeed to be stealthy
                gamma_succ = gamma
                gamma = gamma + step

            else:  # fail
                gamma = gamma - step

            step /= 2

        vector = mean - gamma_succ * std
        vectors = vector.repeat(self.num_byz, 1)

        print(gamma_succ)

        return vectors
