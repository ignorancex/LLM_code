import torch
import time
from .Aggregator import Aggregator


class SelfReject(Aggregator):

    def __init__(self, args):
        super(SelfReject, self).__init__(args)
        self.m = args.num_clients - args.num_byz_resist
        self.server_todo = 'eval'

    def aggregate(self, matrix, origin, eval_func):
        tik = time.time()
        num_client, length = matrix.shape

        losses = []
        for i in range(num_client):
            vector = matrix[i]
            loss, metric, num_data = eval_func(vector)
            losses.append(loss)

        metrics = torch.Tensor(losses)
        indices = torch.argsort(metrics)  # from small to big
        selected_indices = indices[:self.m]

        agg_vector = matrix[selected_indices].mean(dim=0)

        tok = time.time()
        self.running_times.append(tok - tik)

        return agg_vector


class AverageReject(Aggregator):
    def __init__(self, args):
        super(AverageReject, self).__init__(args)
        self.m = args.num_clients - args.num_byz_resist
        self.server_todo = 'eval'

    def aggregate(self, matrix, origin, eval_func):
        tik = time.time()
        num_client, length = matrix.shape

        losses = []
        for i in range(num_client):
            vector = (matrix.sum(axis=0) - matrix[i]) / (num_client - 1)
            loss, metric, num_data = eval_func(vector)
            losses.append(loss)

        metrics = torch.Tensor(losses)
        indices = torch.argsort(metrics, descending=True)  # from big to small
        selected_indices = indices[:self.m]

        agg_vector = matrix[selected_indices].mean(dim=0)

        tok = time.time()
        self.running_times.append(tok - tik)

        return agg_vector


class Zeno(Aggregator):

    def __init__(self, args):
        super(Zeno, self).__init__(args)
        self.m = args.num_clients - args.num_byz_resist
        self.rho = args.zeno_rho
        self.server_todo = 'eval'

    def aggregate(self, matrix, origin, eval_func):
        tik = time.time()
        num_client, length = matrix.shape

        # the first part

        losses = []
        for i in range(num_client):
            vector = matrix[i]
            loss, metric, num_data = eval_func(vector)
            losses.append(loss)

        losses = torch.Tensor(losses)

        # the second part

        updates = matrix - origin
        updates_norms = torch.square(updates.norm(dim=1)).cpu()

        metrics = losses + updates_norms * self.rho

        indices = torch.argsort(metrics)  # from small to big
        selected_indices = indices[:self.m]

        agg_vector = matrix[selected_indices].mean(dim=0)

        tok = time.time()
        self.running_times.append(tok - tik)

        return agg_vector
