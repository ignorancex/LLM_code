import torch
from math import ceil
from scipy.stats import norm
from .Byzantine import Byzantine


class ALittleIsEnough(Byzantine):
    def __init__(self, args):
        super(ALittleIsEnough, self).__init__(args)

        # compute the normal distribution quantile
        p = (ceil((args.num_honest + args.num_byz) / 2) - 1) / args.num_honest
        self.z = norm.ppf(p)  # based on the number of byz and honest clients

    def attack(self, model, matrix):
        num_honest, length = matrix.shape

        # calculate mean and standard deviation

        mean = matrix.mean(dim=0)
        std = matrix.std(dim=0)  # torch.std by default computes the group std, which is an unbiased estimator

        vector = mean - self.z * std
        vectors = vector.repeat(self.num_byz, 1)

        return vectors


class ALittleIsEnough15(Byzantine):
    def __init__(self, args):
        super(ALittleIsEnough15, self).__init__(args)

        # compute the normal distribution quantile
        self.z = 1.5  # based on the number of byz and honest clients

    def attack(self, model, matrix):
        num_honest, length = matrix.shape

        # calculate mean and standard deviation

        mean = matrix.mean(dim=0)
        std = matrix.std(dim=0)  # torch.std by default computes the group std, which is an unbiased estimator

        vector = mean - self.z * std
        vectors = vector.repeat(self.num_byz, 1)

        return vectors