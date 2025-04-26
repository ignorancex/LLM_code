import torch
import torch.nn as nn


class Generator(nn.Module):
    rescale = True

    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.nz = nz
        self.main = nn.ModuleList([
            # input is Z, going into a convolution
            nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
            ),
            # state size. (ngf*8) x 4 x 4
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
            ),
            nn.Sequential(
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
            ),
            # state size. (ngf*2) x 16 x 16
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
            ),
            # state size. (ngf) x 32 x 32
            nn.Sequential(
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
            # state size. (nc) x 64 x 64
        ])
        self._create_input_shapes()

    def _create_input_shapes(self):
        self.input_shapes = []
        z = torch.randn(1, self.nz, 1, 1)
        # Adding empty z2_dim for recover.py
        self.input_shapes.append((tuple((z.shape[1:])), ()))
        for layer in self.main:
            z = layer(z)
            self.input_shapes.append((tuple(z.shape[1:]), ()))

    def forward(self, z, z2=None, n_cuts=0, end=None):
        if end is None:
            end = len(self.main)
        for i, layer in enumerate(self.main[n_cuts:end]):
            z = layer(z)
        return z


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        output = self.main(x)

        return output.view(-1, 1).squeeze(1)
