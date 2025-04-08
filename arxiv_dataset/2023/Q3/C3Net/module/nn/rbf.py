import torch
from torch import nn

class Gaussian_Smearing(nn.Module):
    def __init__(self, start=0, stop=1, n_gaussians=50, trainable=True):
        super(Gaussian_Smearing, self).__init__()
        # compute offset and width of Gaussian functions

        offset = torch.linspace(start, stop, n_gaussians) #[(stop-start)/n_gaussians*i for i in range(n_gaussians)]
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)

    def forward(self, distances):
        coeff = -0.5 / torch.pow(self.width, 2)
        diff = distances[:, :, :, None] - self.offsets[None, None, None, :]

        # compute smear distance values
        gauss = torch.exp(coeff * torch.pow(diff, 2))

        if len(distances.size()) == 4:
            xx
            size = gauss.size()
            gauss = gauss.view(size[0],size[1],size[2],-1)

        return gauss
