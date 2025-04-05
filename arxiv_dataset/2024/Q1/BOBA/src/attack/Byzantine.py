import torch


class Byzantine:
    """
    Base class for a byzantine clients.
    Because Byzantine workers may collude, we use a single agent
    to generate multiple (num_byz) malicious vectors
    """

    def __init__(self, args):
        self.num_byz = args.num_byz
        self.verbose = args.verbose
        self.device = args.device


    def attack(self, model, matrix):
        """
        Only for testing. This function will be overwritten by specific attacks.
        """
        vector = matrix.mean(dim=0)
        vectors = vector.repeat(self.num_byz, 1).to(self.device)
        return vectors
