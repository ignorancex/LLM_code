import torch.nn as nn

class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.z_dim = 100
        ncf = 512
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, ncf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ncf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ncf, ncf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ncf // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ncf // 2, ncf // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ncf // 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ncf // 4, ncf // 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ncf // 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ncf // 8, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.main(z)
        return x

