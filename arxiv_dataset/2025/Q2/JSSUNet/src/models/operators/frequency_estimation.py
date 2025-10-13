from torch import nn
import torch
class GetLowFreq(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            features: int,
            kernel_size: int,
    ):
        super(GetLowFreq, self).__init__()
        self.get_lf=nn.Sequential(*[
                                    nn.Conv2d(in_channels, features, kernel_size=kernel_size, stride=1,
                                              padding=kernel_size // 2, bias=False),
                                    nn.Conv2d(features, out_channels, kernel_size=kernel_size, stride=1,
                                              padding=kernel_size // 2, bias=False),
                                    nn.ReLU()
                                    ])
    def forward(self, u):
        return self.get_lf(u)

class InjectHighFreq(nn.Module):
    def __init__(
            self,
            ms_channels: int,
            hs_channels: int,
            features: int,
            kernel_size: int,
    ):
        super(InjectHighFreq, self).__init__()
        self.get_hf = nn.Sequential(*[nn.Conv2d(ms_channels+hs_channels, features,
                                                        kernel_size=kernel_size, stride=1,
                                                        padding=kernel_size // 2, bias=False),
                                              nn.Conv2d(features,
                                                        features,
                                                        kernel_size=kernel_size, stride=1,
                                                        padding=kernel_size // 2),
                                              nn.Conv2d(features, hs_channels,
                                                        kernel_size=kernel_size, stride=1,
                                                        padding=kernel_size // 2)
                                              ])
        self.relu = nn.ReLU()

    def forward(self, u, ms, hs_up):
        return self.relu(u + self.get_hf(torch.cat([ms, hs_up], dim=1)))