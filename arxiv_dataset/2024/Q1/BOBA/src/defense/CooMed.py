import torch
import time

from .Aggregator import Aggregator


class CooMed(Aggregator):
    """
    Coordinate Median
    """

    def aggregate(self, matrix, origin):
        tik = time.time()
        agg_vector, indices = matrix.median(dim=0)

        tok = time.time()
        self.running_times.append(tok - tik)
        return agg_vector


class TrMean(Aggregator):
    """
    Trimmed Mean
    """

    def __init__(self, args):
        super(TrMean, self).__init__(args)
        self.b = args.num_byz_resist

    def aggregate(self, matrix, origin):
        tik = time.time()
        num_client, length = matrix.shape

        # sort all dimension
        value, indices = torch.sort(matrix, dim=0)

        # get the middle (n - 2b) and average them.
        agg_vector = value[self.b:num_client - self.b].mean(dim=0)

        tok = time.time()
        self.running_times.append(tok - tik)

        return agg_vector
