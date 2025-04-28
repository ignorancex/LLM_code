#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class AnchorLoss(nn.Module):
    r"""Anchor Loss: modulates the standard cross entropy based on
    the prediction difficulty.
        Loss(x, y) = - y * (1 - x + p_pos)^gamma_pos * \log(x)
                        - (1 - y) * (1 + x - p_neg)^gamma_neg * \log(1-x)

        The losses are summed over class and averaged across observations
        for each minibatch.


        Args:
            gamma(float, optional): gamma > 0; reduces the relative loss
            for well-classiﬁed examples,
                                    putting more focus on hard, misclassiﬁed
                                    examples
            slack(float, optional): a margin variable to penalize the output
            variables which are close to
                                    true positive prediction score
            anchor(string, optional): specifies the anchor probability type:
                                      ``pos``: modulate target class loss
                                      ``neg``: modulate background class loss
        Shape:
            - Input: (N, C) where C is the number of classes
            - Target: (N) where each value is the class label of each sample
            - Output: scalar

    """
    def __init__(self, gamma=0.5, slack=0.05, anchor='neg', sigma=2.):
        super(AnchorLoss, self).__init__()

        assert anchor in ['neg', 'pos'], \
            "Anchor type should be either ``neg`` or ``pos``"

        self.gamma = gamma
        self.slack = slack
        self.anchor = anchor
        self.sigma = sigma
        self.sig = nn.Sigmoid()

        if anchor == 'pos':
            self.gamma_pos = gamma
            self.gamma_neg = 0
        elif anchor == 'neg':
            self.gamma_pos = 0
            self.gamma_neg = gamma

    def forward(self, input, target, epoch=None):
        target = target.view(-1, 1)
        pt = input
        logpt_pos = torch.log(input)
        logpt_neg = torch.log(1 - pt)  # log(1-q)

        N = input.size(0)
        C = input.size(1)

        class_mask = input.data.new(N, C).fill_(0)
        class_mask.scatter_(1, target.data, 1.)
        class_mask = class_mask.float()

        pt_pos = pt.gather(1, target).view(-1, 1)
        pt_neg = pt * (1-class_mask)
        pt_neg = pt_neg.max(dim=1)[0].view(-1, 1)
        pt_neg = (pt_neg + self.slack).clamp(max=1).detach()
        pt_pos = (pt_pos - self.slack).clamp(min=0).detach()

        scaling_pos = -1 * (1 - pt + pt_neg).pow(self.gamma_pos)
        loss_pos = scaling_pos * logpt_pos
        scaling_neg = -1 * (1 + pt - pt_pos).pow(self.gamma_neg)
        loss_neg = scaling_neg * logpt_neg

        loss = class_mask * loss_pos + (1 - class_mask) * loss_neg
        loss = loss.sum(1)

        return loss.mean()
