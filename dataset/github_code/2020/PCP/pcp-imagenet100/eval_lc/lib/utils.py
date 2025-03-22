import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable, Function
import torch.nn.functional as F

import models


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


def normalize(data):
    # data in numpy array
    if isinstance(data, np.ndarray):
        row_sums = np.linalg.norm(data, axis=1)
        data = data / row_sums[:, np.newaxis]
        return data

    # data is a tensor
    row_sums = data.norm(dim=1, keepdim=True)
    data = data / row_sums
    return data


class TripletLoss(torch.nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()

        self.margin = margin

    def forward(self, anchor, positive, negative):

        pos_dist = F.pairwise_distance(anchor, positive, 2)
        neg_dist = F.pairwise_distance(anchor, negative, 2)

        hing_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)

        loss = torch.mean(hing_dist)

        return loss


def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

    #     # size of the top layer
    #     N = checkpoint['state_dict']['top_layer.bias'].size()
    #
    #     # build skeleton of the model
    #     sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
    #     model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))
    #
    #     # deal with a dataparallel table
    #     def rename_key(key):
    #         if not 'module' in key:
    #             return key
    #         return ''.join(key.split('.module'))
    #
    #     checkpoint['state_dict'] = {rename_key(key): val
    #                                 for key, val
    #                                 in checkpoint['state_dict'].items()}
    #
    #     # load weights
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print("Loaded")
    # else:
    #     model = None
    #     print("=> no checkpoint found at '{}'".format(path))

        # net = models.__dict__['ResNet18'](low_dim=128)
        # net = models.__dict__['resnet18'](low_dim=128)

        net = models.__dict__['alexnet'](out=128)
        # net = models.__dict__['Alexnet_C'](out=args.low_dim)

        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        net.load_state_dict(checkpoint['net'])

    return net