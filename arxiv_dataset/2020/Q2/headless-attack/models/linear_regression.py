'''
Polynomial regression with interactions in PyTorch.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegression(nn.Module):
  def __init__(self, numfeat):
    super(LinearRegression, self).__init__()
    self.fc = nn.Linear(numfeat, 10, bias=False)

  def forward(self, x):
    out = self.fc(x)  # linear layer
    return out
