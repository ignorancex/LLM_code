import torch
import time

from .Aggregator import Aggregator


class Average(Aggregator):
    """
    Unweighted Average
    """

    def aggregate(self, matrix, origin):
        tik = time.time()
        agg_vector = matrix.mean(dim=0)
        tok = time.time()
        self.running_times.append(tok - tik)
        return agg_vector
