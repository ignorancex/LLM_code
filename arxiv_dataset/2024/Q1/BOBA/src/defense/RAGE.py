import torch
from sklearn.decomposition import TruncatedSVD
import time

from .Aggregator import Aggregator


class RAGE(Aggregator):
    """
    Krum
    """

    def __init__(self, args):
        super(RAGE, self).__init__(args)
        self.k = args.num_clients - args.num_byz_resist - 1
        self.max_iter = args.rage_max_iter
        self.svd = TruncatedSVD(n_components=1)

    def aggregate(self, matrix, origin):
        tik = time.time()
        num_client, length = matrix.shape

        updates = (matrix - origin).cpu()

        w = torch.ones(num_client) / num_client

        for it in range(self.max_iter):

            mu = (updates * w.view(-1, 1)).sum(dim=0) / w.sum()
            Sigma_sqrt = torch.sqrt(w / w.sum()).view(-1, 1) * (updates - mu)

            Sigma_sqrt = Sigma_sqrt.numpy()
            self.svd.fit(Sigma_sqrt)

            pc = self.svd.components_[0]
            pc = torch.Tensor(pc)

            tau = torch.square(updates @ pc)
            tau_max = (tau[w > 0]).max()
            w = (1 - tau / tau_max) * w

            # print(w / w.sum())

        update = (updates * w.view(-1, 1)).sum(dim=0) / w.sum()
        update = update.to(self.device)

        agg_vector = origin + update

        tok = time.time()
        self.running_times.append(tok - tik)

        return agg_vector

