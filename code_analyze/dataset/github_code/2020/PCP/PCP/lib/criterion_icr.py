import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
# modify criterion_icr

class ICRcriterion(nn.Module):
    """
        Loss of ICR (Instance and Cluster Recognition)
        most reference AND_ans_criterion.py
    """

    def __init__(self):
        super(ICRcriterion, self).__init__()

    def forward(self, x, y, ICRs):
        batch_size, _ = x.shape

        # split cluster and instance list
        cluster_index, instance_index = self.__split(y, ICRs)
        preds = F.softmax(x, 1)

        l_cls = 0.
        if cluster_index.size(0) > 0:
            # compute loss for cluster samples
            y_cls = y.index_select(0, cluster_index)
            y_cls_neighbour = ICRs.position.index_select(0, y_cls)
            # p_i = \sum_{j \in \Omega_i} p_{i,j}
            x_cls = preds.index_select(0, cluster_index)

            idx = 0
            x_cls_neighbopur = torch.zeros(y_cls_neighbour.shape[0]).float().cuda()
            for i in y_cls_neighbour:
                x_cls_neighbopur[idx] = x_cls[idx].gather(0, ICRs.neighbours[i.data.item()]).sum()
                idx += 1
            x_cls = x_cls.gather(1, y_cls.view(-1, 1)).view(-1) + x_cls_neighbopur
            # NLL: l = -log(p_i)
            l_cls = -1 * torch.log(x_cls).sum(0)

        l_ins = 0.
        if instance_index.size(0) > 0:
            # compute loss for instance samples
            y_ins = y.index_select(0, instance_index)
            x_ins = preds.index_select(0, instance_index)
            # p_i = p_{i, i}
            x_ins = x_ins.gather(1, y_ins.view(-1, 1))
            # NLL: l = -log(p_i)
            l_ins = -1 * torch.log(x_ins).sum(0)

        return (l_ins + l_cls) / batch_size

    def __split(self, y, ICRs):
        """
        split current batch index (y) to instance index and cluster index
        """
        pos = ICRs.position.index_select(0, y.view(-1))

        return (pos >= 0).nonzero().view(-1), (pos < 0).nonzero().view(-1)

