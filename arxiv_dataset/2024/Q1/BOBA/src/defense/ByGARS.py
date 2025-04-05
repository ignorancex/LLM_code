import torch
import torch.nn.functional as F
import time

from .Aggregator import Aggregator


class ByGARS(Aggregator):

    def __init__(self, args):
        super(ByGARS, self).__init__(args)
        self.lr = args.lm_lr  # current learning rate

        self.k = args.bygars_k
        self.alpha = args.bygars_alpha

        self.server_todo = 'train'


    def aggregate(self, matrix, origin, train_func):
        tik = time.time()

        client_updates = matrix - origin

        client_norm_grads = - client_updates / client_updates.norm(dim=1, keepdim=True)

        # initialize q with zeros
        self.q = torch.zeros(matrix.shape[0]).to(self.device)

        # optimize q
        for i in range(self.k):
            # line 7
            pred_grad = (client_norm_grads * self.q.view(-1, 1)).sum(dim=0)
            pred_vector = origin - self.lr * pred_grad

            # line 8
            new_vector = train_func(pred_vector)
            new_update = new_vector - pred_vector
            new_norm_grad = - new_update / new_update.norm()

            # update q
            self.q += self.alpha * self.lr * (client_norm_grads @ new_norm_grad)

        # update w
        agg_grad = (client_norm_grads * self.q.view(-1, 1)).sum(dim=0)

        agg_vector = origin - self.lr * agg_grad

        tok = time.time()
        self.running_times.append(tok - tik)

        return agg_vector


