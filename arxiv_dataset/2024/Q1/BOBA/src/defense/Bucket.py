import torch
import time

from .Aggregator import Aggregator
from .Krum import Krum, MultiKrum


def bucket(matrix, s):
    num_client, length = matrix.shape
    num_vector = num_client // s
    num_raw_vector = num_vector * s
    indices = torch.randperm(num_client)[:num_raw_vector].view(num_vector, s)
    bucket_matrix = matrix[indices].mean(dim=1)

    return bucket_matrix


class BucketKrum(Krum):

    def __init__(self, args):
        Aggregator.__init__(self, args)
        self.s = args.bucket_s
        self.num_vector = args.num_clients // self.s
        self.k = self.num_vector - args.num_byz_resist - 1

    def aggregate(self, matrix, origin):

        tik = time.time()

        bucket_matrix = bucket(matrix, self.s)
        agg_vector = Krum.aggregate(self, bucket_matrix, origin)

        tok = time.time()
        self.running_times.append(tok - tik)

        return agg_vector


class BucketMultiKrum(MultiKrum):

    def __init__(self, args):
        Aggregator.__init__(self, args)
        self.s = args.bucket_s
        self.num_vector = args.num_clients // self.s
        self.k = self.num_vector - args.num_byz_resist - 1
        self.m = self.num_vector - args.num_byz_resist

    def aggregate(self, matrix, origin):
        tik = time.time()

        bucket_matrix = bucket(matrix, self.s)
        agg_vector = MultiKrum.aggregate(self, bucket_matrix, origin)

        tok = time.time()
        self.running_times.append(tok - tik)

        return agg_vector
