import torch
import torch.nn.functional as F
import time

from .Aggregator import Aggregator


class FLTrust(Aggregator):
    def __init__(self, args):
        super(FLTrust, self).__init__(args)
        self.server_todo = 'vector'

    def aggregate(self, matrix, origin, server_vector):
        tik = time.time()
        server_update = server_vector - origin

        client_updates = matrix - origin

        TS = F.relu(F.cosine_similarity(client_updates, server_update.view(1, -1)))  # compute TS

        TS = TS / TS.sum()  # normalize

        client_norm_updates = client_updates / torch.norm(client_updates, dim=1, keepdim=True) * torch.norm(server_update)

        agg_update = (client_norm_updates * TS.view(-1, 1)).sum(dim=0)

        agg_vector = origin + agg_update
        tok = time.time()
        self.running_times.append(tok - tik)

        return agg_vector

