import torch
import torch.nn as nn
import numpy as np

class RCE(nn.Module):
    def __init__(self, gamma_val=1.7):
        super(RCE, self).__init__()
        self.gamma_val = gamma_val

    def forward(self, img):
        img = img.to(torch.float64)

        f_ij = (1 / np.sqrt(2 * np.pi)) * torch.exp(-0.5 * img ** 2)

        s_ij = torch.log(1 + torch.exp(img))

        l_ij = torch.sqrt(f_ij + s_ij + f_ij * s_ij)

        l_min = torch.min(l_ij)
        l_max = torch.max(l_ij)
        n_ij = (l_ij - l_min) / (l_max - l_min)

        n_ij_gamma = torch.pow(n_ij, self.gamma_val)

        return n_ij_gamma