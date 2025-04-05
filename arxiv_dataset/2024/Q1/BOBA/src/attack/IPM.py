import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .Byzantine import Byzantine


class IPM(Byzantine):
    def __init__(self, args):
        super(IPM, self).__init__(args)
        self.factor = args.signflip_factor

    def attack(self, model, matrix):
        old_vector = model.get_params_tensor()
        new_vector = matrix.mean(dim=0)

        update = new_vector - old_vector

        flipped_vector = old_vector + self.factor * update

        vectors = flipped_vector.repeat(self.num_byz, 1)

        return vectors
