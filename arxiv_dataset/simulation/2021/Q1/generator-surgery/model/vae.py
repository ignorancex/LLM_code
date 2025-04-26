# Based off https://github.com/pytorch/examples/blob/master/vae/main.py
# https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
# https://github.com/podgorskiy/VAE

import torch
import torch.utils.data
from torch import nn


class DownConv(nn.Module):
    def __init__(self, cin, cout, act):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, 2, 1)
        self.bn = nn.BatchNorm2d(cout)
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.act = nn.LeakyReLU(inplace=True)
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList([
            DownConv(3, 128, self.act),
            DownConv(128, 256, self.act),
            DownConv(256, 512, self.act),
            DownConv(512, 1024, self.act),
            DownConv(1024, 2048, self.act),
        ])

        self.mu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 4 * 4, self.latent_dim),
        )
        self.logvar = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 4 * 4, self.latent_dim),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.mu(x), self.logvar(x)


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class UpConv(nn.Module):
    def __init__(self, cin, cout, act):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(cin, cout, 3, 1, 1)
        self.bn = nn.BatchNorm2d(cout)
        self.act = act

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Decoder(nn.Module):
    rescale = False

    def __init__(self, latent_dim):
        super().__init__()
        self.act = nn.LeakyReLU(inplace=True)
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim, 2048 * 4 * 4),
                View((-1, 2048, 4, 4)),
            ),
            UpConv(2048, 1024, self.act),
            UpConv(1024, 512, self.act),
            UpConv(512, 256, self.act),
            UpConv(256, 128, self.act),
            UpConv(128, 128, self.act),
            nn.Sequential(
                nn.Conv2d(128, 3, kernel_size=3, padding=1),
                nn.Sigmoid(),
            ),
        ])
        self._get_input_shapes()

    def _get_input_shapes(self):
        z = torch.randn(1, self.latent_dim)
        self.input_shapes = []
        self.input_shapes.append((tuple(z.shape[1:]), ()))
        for layer in self.layers:
            z = layer(z)
            self.input_shapes.append((tuple(z.shape[1:]), ()))

    def forward(self, z1, z2=None, n_cuts=0, end=None):
        if end is None:
            end = len(self.layers)
        for layer in self.layers[n_cuts:end]:
            z1 = layer(z1)
        return z1


class VAE(nn.Module):
    def __init__(self, latent_dim=512, img_shape=(3, 128, 128)):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
