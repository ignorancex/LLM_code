import torch
from torch.nn import Module
from geomloss import SamplesLoss

class SinkhornLoss(Module):
    

    def __init__(self, blur=0.3, scaling=.8):
        super(SinkhornLoss, self).__init__()
        self.loss = SamplesLoss("sinkhorn", blur=blur, scaling=scaling)
        
    def forward(self, *args):
        x, y = args
        x_f = torch.flatten(x, start_dim=2, end_dim=3)
        y_f = torch.flatten(y, start_dim=2, end_dim=3)
        return torch.mean(self.loss(x_f, y_f))
    