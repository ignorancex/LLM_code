from sympy import factorint
from torch import nn

from src.models.operators.downsampling import Downsampling


class Up(nn.Module):
    def __init__(self, channels, sampling,  **kwargs ):
        super(Up, self).__init__()
        self.sampling = sampling
        up_layers = []
        for p, exp in factorint(sampling).items():
            for _ in range(exp):
                kernel = p + 1 if p % 2 == 0 else p + 2

                up_layers.append(nn.ConvTranspose2d(in_channels=channels,
                                                    out_channels=channels,
                                                    kernel_size=kernel,
                                                    stride=p,
                                                    padding=kernel // 2,
                                                    bias=False,
                                                    output_padding=p - 1,
                                                   ))
                kernel = 2*p+1
                up_layers.append(nn.Conv2d(in_channels=channels,
                                                   out_channels=channels,
                                                   kernel_size=kernel,
                                                   padding=kernel // 2,
                                                   bias=False
                                                   ))
        self.up = nn.Sequential(*up_layers)
    def forward(self, input):
        return self.up(input)


class DBPUp(nn.Module):
    def __init__(self, channels, sampling, iter,  **kwargs):
        super(DBPUp, self).__init__()
        self.iter = iter
        self.Up0 = Up(channels, sampling)
        self.Down = nn.ModuleList([Downsampling(channels, sampling,groups=1) for _ in range(iter)])
        self.Up = nn.ModuleList([Up(channels, sampling) for _ in range(iter)])
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        low_pre = x
        x = self.relu(self.Up0(x))
        for i in range(self.iter):
            x_pre = x.clone()
            low = self.relu(self.Down[i](x))
            x = self.relu(self.Up[i](low-low_pre))
            low_pre = low
            x = x_pre + x
        return x