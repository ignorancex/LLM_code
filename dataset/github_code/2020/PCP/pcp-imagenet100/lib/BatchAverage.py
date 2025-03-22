import torch
from torch.autograd import Function
from torch import nn
import math
import numpy as np

# ref CVPR2019 Unsupervised_Embedding_Learning


class BatchCriterion(nn.Module):
    ''' Compute the loss within each batch
    '''

    def __init__(self, negM, T, batchSize, ndata):
        super(BatchCriterion, self).__init__()
        self.negM = negM
        self.T = T
        self.diag_mat = 1 - torch.eye(batchSize * 2).cuda()
        self.diag_mat_lastbatch = 1 - torch.eye((ndata-(ndata//batchSize * batchSize)) * 2).cuda()
        self.ndata = ndata
        self.batch_size = batchSize

    def forward(self, x, targets):
        batchSize = x.size(0)

        # get positive innerproduct
        reordered_x = torch.cat((x.narrow(0, batchSize // 2, batchSize // 2), x.narrow(0, 0, batchSize // 2)), 0)
        # reordered_x = reordered_x.data
        # pos = (x * reordered_x.data).sum(1).div_((x * reordered_x.data).sum(1).max()).div_(self.T).exp_()
        pos = (x * reordered_x.data).sum(1).div_(self.T).exp_()

        # get all innerproduct, remove diag
        if batchSize == self.batch_size * 2:

            # all_prob = torch.mm(x, x.t().data).div_(torch.mm(x, x.t().data).max()).div_(self.T).exp_() * self.diag_mat
            all_prob = torch.mm(x, x.t().data).div_(self.T).exp_() * self.diag_mat
        else:
            # print(batchSize) # 384
            # all_prob = torch.mm(x, x.t().data).div_(torch.mm(x, x.t().data).max()).div_(self.T).exp_() * self.diag_mat_lastbatch
            all_prob = torch.mm(x, x.t().data).div_(self.T).exp_() * self.diag_mat_lastbatch

        # all_prob = torch.mm(x, x.t().data).div_(self.T).exp_() * self.diag_mat

        if self.negM == 1:
            all_div = all_prob.sum(1)
        else:
            # remove pos for neg
            all_div = (all_prob.sum(1) - pos) * self.negM + pos

        lnPmt = torch.div(pos, all_div)

        # negative probability
        Pon_div = all_div.repeat(batchSize, 1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)

        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        lnPonsum = lnPonsum * self.negM
        loss = - (lnPmtsum + lnPonsum) / batchSize
        return loss
