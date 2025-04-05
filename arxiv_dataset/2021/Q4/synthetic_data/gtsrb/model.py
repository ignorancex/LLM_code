# code adapted from PyTorch DCGAN tutorial
# https://github.com/pytorch/examples/tree/master/dcgan

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, nz=100, nef=64 * 2, nc=3):
        super().__init__()
        self.module = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # state size. (nef) x 32 x 32
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2),
            # state size. (nef*2) x 16 x 16
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2),
            # state size. (nef*4) x 8 x 8
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.2),
            # state size. (nef*8) x 4 x 4
            nn.Conv2d(nef * 8, nz, 4, 1, 0, bias=False)
        )

    def forward(self, inp):
        return self.module(inp)


class Classify(nn.Module):
    def __init__(self, classes=None, nz=100):
        super(Classify, self).__init__()
        assert classes != None, "Must provide number of target classes"

        model = [nn.Flatten()]
        while nz // 4 > 2 * classes:
            model.append(nn.Linear(nz, nz // 4, bias=False))
            model.append(nn.ReLU())
            nz //= 4
        model.append(nn.Linear(nz, classes, bias=False))

        self.classify = nn.Sequential(*model)

    def forward(self, inp):
        return self.classify(inp)


class Classifier(nn.Module):
    def __init__(self, classes=43):
        super().__init__()
        self.encoder = Encoder()
        self.classify = Classify(classes=classes)

    def forward(self, inp, **kwargs):
        out = self.encoder(inp)
        return self.classify(out)

