import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ICRDiscovery(nn.Module):
    """
    Instance and Cluster Recognition
    Args:
        nsamples: total number of samples, e.g. cifar10 - 50k, ImageNet - 1300k
    Var:
        position: indicate instance level or cluster level
        neighbours: instances belong to same cluster
    """

    def __init__(self, nsamples):
        super(ICRDiscovery, self).__init__()
        self.samples_num = nsamples
        self.position = -1 * torch.arange(nsamples).long().cuda() - 1
        d = dict()
        self.neighbours = d

    def forward(self):

        return None

    def update(self):

        return None

# A = ICRDiscovery(3)
# print(A.samples_num)
# A.samples_num = 5
# print(A.samples_num)
