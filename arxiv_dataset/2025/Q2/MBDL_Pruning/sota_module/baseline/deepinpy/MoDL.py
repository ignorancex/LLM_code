from torch import nn
import torch
from .conjgrad import ConjGrad
from einops import rearrange


class MoDLBlock(nn.Module):
    def __init__(self, mu_init, cg_max_iter, channels):
        super().__init__()

        self.net = ResNet5Block(num_filters_start=channels, num_filters_end=channels)
        self.l2lam = torch.nn.Parameter(torch.ones(1) * mu_init)
        self.cg_max_iter = cg_max_iter

    def forward(self, x, AHA, inp):

        r = self.net(x)

        cg_op = ConjGrad(inp + self.l2lam * r, AHA, l2lam=self.l2lam, max_iter=self.cg_max_iter, verbose=False)
        x = cg_op.forward(x)

        return x


class MoDL(nn.Module):
    def __init__(self, mu_init, cg_max_iter, iteration, channels):
        super().__init__()

        self.iteration = iteration
        self.modl_block = MoDLBlock(
            mu_init=mu_init,
            cg_max_iter=cg_max_iter,
            channels=channels,
        )

    def forward(self, x, AHA):

        inp = x

        for _ in range(self.iteration):
            x = self.modl_block(x, AHA, inp)

        return x


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class ResNet5Block(torch.nn.Module):
    def __init__(self, num_filters=64, filter_size=3, num_filters_start=2, num_filters_end=2, batch_norm=True):
        super(ResNet5Block, self).__init__()

        if batch_norm:
            self.model = torch.nn.Sequential(
                Conv2dSame(num_filters_start, num_filters, filter_size),
                torch.nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                Conv2dSame(num_filters, num_filters, filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters, num_filters, filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters, num_filters, filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters, num_filters, filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters, num_filters_end, filter_size)
            )
        else:
            self.model = torch.nn.Sequential(
                Conv2dSame(num_filters_start, num_filters, filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters, num_filters, filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters, num_filters, filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters, num_filters, filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters, num_filters, filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters, num_filters_end, filter_size)
            )

    def forward(self, x):
        return x + self.step(x)

    def step(self, x):

        if x.dtype == torch.complex64:
            x_hat = torch.view_as_real(x)
            x_hat = rearrange(x_hat, 'b w h c -> b c w h')

            x_hat = self.model(x_hat)

            x_hat = rearrange(x_hat, 'b c w h -> b w h c')
            x_hat = x_hat[..., 0] + x_hat[..., 1] * 1j

        elif x.dtype == torch.float32 and x.dim() == 3:
            x_hat = self.net(x.unsqueeze(1)).squeeze(1)

        else:
            raise NotImplementedError()

        return x_hat
