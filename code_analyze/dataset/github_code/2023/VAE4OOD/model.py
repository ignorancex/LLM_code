import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import utils

# this structure is based on https://arxiv.org/pdf/1810.01392.pdf
# to use for fair comparisons 
base = 32
class Encoder(nn.Module):
    def __init__(self, shape, nz):
        super(Encoder, self).__init__()
        self.nz = nz
        if shape[0] == 28: # img size 28x28
            c = 2 # used for fully connected layer build
        elif shape[0] == 32: # img size 30x30
            c = 8

        self.conv = nn.Sequential(
            nn.Conv2d(shape[-1], base, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(base, base, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(base, 2*base, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(2*base, 2*base, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(2*base, 2*nz, kernel_size=7, stride=1, padding=0),
            nn.LeakyReLU()
        )

        self.lin1 = nn.Linear(c*nz, nz)
        self.lin2 = nn.Linear(c*nz, nz)

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, x):
        x = self.conv(x)
        
        s = x.shape
        x = torch.reshape(x, (s[0], np.prod(s[1:])))

        mu = self.lin1(x)
        logvar = self.lin2(x)

        
        z = self.reparametrize(mu,logvar)
        return [z, mu, logvar]
 
class Decoder(nn.Module):
    def __init__(self, shape, nz, f):
        super(Decoder, self).__init__()
        self.shape = shape
        self.f = f

        self.deconv = nn.Sequential( # nz
            nn.ConvTranspose2d(nz, 2*base, kernel_size=shape[0]//4, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2*base, 2*base, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2*base, 2*base, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2*base, base, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(base, base, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(base, base, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(base, f*shape[-1], kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[1], 1, 1))
        
        x = self.deconv(x)
        if self.f == 1:
            return x
        else:
            x = x.view(-1, self.shape[2], 256, self.shape[0], self.shape[1])
            x = x.permute(0, 1, 3, 4, 2)
            return x


