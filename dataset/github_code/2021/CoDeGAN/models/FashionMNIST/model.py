import torch.nn as nn
import torch.nn.functional as F


# Weight Initializer
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class Generator(nn.Module):
    def __init__(self, zn_dim, zc_dim):
        super(Generator, self).__init__()
        self.zn_dim = zn_dim
        self.zc_dim = zc_dim
        self.fc = nn.Sequential(
            nn.Linear(self.zc_dim + self.zn_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        initialize_weights(self)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)
        x = self.cnn(x)
        return x


class Encoder(nn.Module):
    def __init__(self, zn_dim, fc_dim):
        super(Encoder, self).__init__()
        self.zn_dim = zn_dim
        self.fc_dim = fc_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.fc_dim + self.zn_dim),
        )
        initialize_weights(self)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc(x)
        zn = x[:, : self.zn_dim]
        zc = x[:, self.zn_dim :]
        zc = F.normalize(zc, dim=1)
        return zn, zc


class Discriminator(nn.Module):
    def __init__(self, wgan_gp=False):
        super(Discriminator, self).__init__()
        self.wgan_gp = wgan_gp
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )
        if not self.wgan_gp:
            self.fc = nn.Sequential(self.fc, nn.Sigmoid())
        initialize_weights(self)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc(x)
        return x
