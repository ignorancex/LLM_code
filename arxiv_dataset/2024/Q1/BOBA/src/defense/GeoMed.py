import torch
import numpy as np
from hdmedians import geomedian
import time

from .Aggregator import Aggregator


class GeoMed(Aggregator):
    """
    Geometric Median
    """

    def aggregate(self, matrix, origin):
        print('calling geomed')
        tik = time.time()
        agg_vector = torch.Tensor(geomedian(matrix.cpu().numpy(), axis=0)).to(self.device)
        tok = time.time()
        self.running_times.append(tok - tik)
        return agg_vector
