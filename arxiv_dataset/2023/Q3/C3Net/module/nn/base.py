import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as functional
from functools import partial

class Dense(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, activation=None):
        self.weight_init = xavier_uniform_
        self.bias_init = partial(constant_, val=0.0)
        self.activation = activation
        super(Dense, self).__init__(in_features, out_features, bias)
    
    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)
    
    def forward(self, inputs):
        y = super(Dense, self).forward(inputs)
        if self.activation:
            y = functional.relu(y)
        return y

class GetItem(nn.Module):
    def __init__(self, key):
        super(GetItem, self).__init__()
        self.key = key

    def forward(self, inputs):
        return inputs[self.key]

class Aggregate(nn.Module):
    def __init__(self, axis):
        super(Aggregate, self).__init__()
        self.axis = axis

    def forward(self, input, mask=None):
        if mask is not None:
            input = input * mask[..., None]

        y = torch.sum(input, self.axis)
        
        return y
