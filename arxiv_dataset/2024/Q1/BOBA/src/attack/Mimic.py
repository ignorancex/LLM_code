import torch
from torch.distributions import Normal
from .Byzantine import Byzantine


class Mimic(Byzantine):
    """
    Mimic the behavior of one client.
    To maximize the cumulative effects, we always copy the first client.
    """

    def attack(self, model, matrix):
        vector = matrix[0]
        vectors = vector.repeat(self.num_byz, 1)

        return vectors
