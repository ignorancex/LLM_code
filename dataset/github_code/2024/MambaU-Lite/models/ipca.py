import torch.nn as nn

import torch
import torch.nn.functional as F
from einops import rearrange, reduce

class PSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.prob = nn.Softmax2d()

    def forward(self, x):
        # Compute the spatial mean of the input
        s = reduce(x, 'b c w h -> b w h', 'mean')

        # Apply pointwise convolution
        x = self.pw(x)

        # Compute the spatial mean after convolution
        s_ = reduce(x, 'b c w h -> b w h', 'mean')

        # Compute the attention scores
        raise_sp = self.prob(s_ - s)
        att_score = torch.sigmoid(s_ * (1 + raise_sp))

        # Ensure dimensions match correctly for multiplication
        att_score = att_score.unsqueeze(1)  # Shape [batch_size, 1, height, width]
        return x * att_score  # Broadcasting to match the dimensions


# Priority Channel Attention (PCA)
class PCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=9, groups=dim, padding="same")
        self.prob = nn.Softmax(dim=1)

    def forward(self, x):
        # Compute the channel-wise mean of the input
        c = reduce(x, 'b c w h -> b c', 'mean')

        # Apply depthwise convolution
        x = self.dw(x)

        # Compute the channel-wise mean after convolution
        c_ = reduce(x, 'b c w h -> b c', 'mean')

        # Compute the attention scores
        raise_ch = self.prob(c_ - c)
        att_score = torch.sigmoid(c_ * (1 + raise_ch))

        # Ensure dimensions match correctly for multiplication
        att_score = att_score.unsqueeze(2).unsqueeze(3)  # Shape [batch_size, channels, 1, 1]
        return x * att_score  # Broadcasting to match the dimensions

# The structure of BottleNeck in U-shape model, named IPCA.
class BottleneckPCAPSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pca = PCA(dim)
        self.psa = PSA(dim)
        #self.res= Residual()
    def forward(self, x):
        # Apply PCA with residual connection
        pca_out=self.pca(x)

        # Apply PSA
        psa_out = self.psa(pca_out)

        return psa_out
