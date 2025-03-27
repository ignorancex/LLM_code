from orthogonal.ortho import orthogonal_with_density
import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarConv(nn.Module):
    def __init__(self, output=10, depth=7, actv_fn="tanh", last="logits", init_weights=None, width=128):
        super(CifarConv, self).__init__()
        
        if actv_fn == "tanh":
            actv_fn  = nn.Tanh
        elif actv_fn == "linear":
            actv_fn = nn.Identity
        elif actv_fn == "relu":
            actv_fn = nn.ReLU
        elif actv_fn == "selu":
            actv_fn = nn.SELU
        elif actv_fn == "hard_tanh":
            actv_fn = nn.Hardtanh
        else:
            raise ValueError("Unknown activation")
        self.actv_fn = actv_fn
        
        assert depth>=1, "Depth must be at least one"
        self.depth = depth
        self.last = last
        self.width = width

        layers =[]

        # 32 x 32
        layers.append(nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, padding_mode='circular', padding=1))
        layers.append(self.actv_fn())

        layers.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding_mode='circular', padding=1, stride=2))
        layers.append(self.actv_fn())
        # 16 x 16
        layers.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding_mode='circular', padding=1, stride=2))
        layers.append(self.actv_fn())
        # 8 x 8

        for i in range(depth - 6):
            layers.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding_mode='circular', padding=1))
            layers.append(self.actv_fn())
        
        layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(width * 4 * 4, output))

        self.net = nn.Sequential(*layers)

        if init_weights is not None:
            self.apply(init_weights)

    def forward(self, x):
        out = self.net(x)
        if self.last == "logsoftmax":
            return F.log_softmax(out, dim=1)
        elif self.last == "logits":
            return out
        else:
            raise ValueError("Unknown last operation")
        
