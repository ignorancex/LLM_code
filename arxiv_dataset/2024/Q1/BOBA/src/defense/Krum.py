import torch
import time

from .Aggregator import Aggregator


class Krum(Aggregator):
    """
    Krum
    """

    def __init__(self, args):
        super(Krum, self).__init__(args)
        self.k = args.num_clients - args.num_byz_resist - 1

    def aggregate(self, matrix, origin):
        tik = time.time()
        num_client, length = matrix.shape

        # compute pairwise distance
        dist = torch.zeros(size=(num_client, num_client))

        # to avoid OOM, we use for loop.
        for i in range(num_client):
            dist[i] = torch.square(torch.norm(matrix - matrix[i], dim=1))

        # compute score and select the lowest one
        scores = torch.sort(dist, dim=1)[0][:, :self.k].sum(dim=1)
        indices = torch.argsort(scores)
        selected_index = indices[0]

        agg_vector = matrix[selected_index]
        # print(selected_index)
        tok = time.time()

        self.running_times.append(tok - tik)

        return agg_vector


class MultiKrum(Aggregator):
    """
    Multi Krum
    """

    def __init__(self, args):
        super(MultiKrum, self).__init__(args)
        self.k = args.num_clients - args.num_byz_resist - 1
        self.m = args.num_clients - args.num_byz_resist

    def aggregate(self, matrix, origin):
        tik = time.time()
        num_client, length = matrix.shape

        # compute pairwise distance
        dist = torch.zeros(size=(num_client, num_client))
        for i in range(num_client):
            dist[i] = torch.square(torch.norm(matrix - matrix[i], dim=1))

        # compute score and select the lowest ones
        scores = torch.sort(dist, dim=1)[0][:, :self.k].sum(dim=1)
        indices = torch.argsort(scores)
        selected_indices = indices[:self.m]

        agg_vector = matrix[selected_indices].mean(dim=0)

        tok = time.time()
        self.running_times.append(tok - tik)

        return agg_vector
