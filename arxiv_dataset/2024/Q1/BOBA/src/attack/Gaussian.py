import torch
from torch.distributions import Normal
from .Byzantine import Byzantine


class Gaussian(Byzantine):
    """
    Send malicious updates that follows Gaussian distribution.
    """

    def __init__(self, args):
        super(Gaussian, self).__init__(args)
        self.std = args.gaussian_std
        self.dist = Normal(loc=0.0, scale=self.std)
        self.collude = args.gaussian_collude

    def attack(self, model, matrix):

        old_vector = model.get_params_tensor()
        length = old_vector.shape[0]

        if self.collude:
            noise = self.dist.sample((length,)).to(self.device)
            vector = old_vector + noise
            vectors = vector.repeat(self.num_byz, 1)
        else:
            noise = self.dist.sample((self.num_byz, length)).to(self.device)
            vectors = old_vector + noise

        return vectors
