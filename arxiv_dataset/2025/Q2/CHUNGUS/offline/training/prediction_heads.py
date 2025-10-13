import torch
import torch.nn as nn


class BasicPredictionHead(nn.Module):
    """ Basic prediction head """
    def __init__(self, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(emb_dim, 128, kernel_size=1, padding=0, stride=1, bias=True)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 1, kernel_size=1, padding=0, stride=1, bias=True)
        self.act2 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return {'prediction': x}


class ReconstructionBasicPredictionHead(nn.Module):
    """ Prediction head that includes reconstruction branch """
    def __init__(self, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(emb_dim, 128, kernel_size=1, padding=0, stride=1, bias=True)
        self.act1 = nn.ReLU()
        self.convP = nn.Conv2d(128, 1, kernel_size=1, padding=0, stride=1, bias=True)
        self.actP = nn.Sigmoid()
        self.convR = nn.Conv2d(128, emb_dim, kernel_size=1, padding=0, stride=1, bias=True)
    
    def forward(self, x):
        enc = self.act1(self.conv1(x))
        p = self.actP(self.convP(enc))
        r = torch.square(self.convR(enc) - x).mean(dim=1, keepdims=True)
        return {'prediction': p, 'reconstruction': r}


def get_args(prefix, name):
    """ Get arguments from a prefix and name """
    str_args = name.replace(prefix, '').split('/')[1:]
    args = {}
    for s in str_args:
        key, val = s.split(':')
        if val.isdigit():
            val = int(val)
        elif val.replace('.','',1).isdigit() and val.count('.') < 2:
            val = float(val)
        else:
            val = str(val) # this is redundant
        args[key] = val
    return args


def get_head(name, **kwargs):
    """ Get a prediction head from a string """
    if name == "basic":
        return BasicPredictionHead(**kwargs)
    elif name == "reconstructionbasic":
        return ReconstructionBasicPredictionHead(**kwargs)
    else:
        raise ValueError("Invalid prediction head")
