import torch.nn as nn
import torch.nn.functional as F

"""
Code for the model acquired from https://github.com/yl-1993/ConvDeltaOrthogonal-Init
"""


__all__ = [
    "van4",
    "van8",
    "van12",
    "van16",
    "van32",
    "van128",
    "van256",
    "van512",
    "van768",
    "van1024",
]


class Vanilla(nn.Module):
    def __init__(self, base, c, num_classes=10, last="logits", init_weights=None):
        super(Vanilla, self).__init__()
        self.base = base
        self.fc = nn.Linear(c * 4, num_classes)
        self.last = last
        if init_weights is not None:
            self.apply(init_weights)

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.last == "logsoftmax":
            return F.log_softmax(x, dim=1)
        elif self.last == "logits":
            return x
        else:
            raise ValueError("Unknown last operation")



def make_layers(depth, c, activation):
    assert isinstance(depth, int)

    if activation == "tanh":
        act = nn.Tanh()
    elif activation == "relu":
        act = nn.ReLU()

    layers = []
    in_channels = 3
    for stride in [1, 2, 2]:
        conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1, stride=stride)
        layers += [conv2d, act]
        in_channels = c
    for i in range(depth):
        if i == depth // 2 - 1:
            conv2d = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=2)
        elif i > depth - 2:
            conv2d = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=2)

        else:
            conv2d = nn.Conv2d(c, c, kernel_size=3, padding=1)
        layers += [conv2d, act]
    return nn.Sequential(*layers), c


def van4(c, activation, **kwargs):
    """Constructs a 8 layers vanilla model."""
    model = Vanilla(*make_layers(4, c, activation), **kwargs)
    return model


def van8(c, activation, **kwargs):
    """Constructs a 8 layers vanilla model."""
    model = Vanilla(*make_layers(8, c, activation), **kwargs)
    return model


def van12(c, activation, **kwargs):
    """Constructs a 12 layers vanilla model."""
    model = Vanilla(*make_layers(12, c, activation), **kwargs)
    return model


def van16(c, activation, **kwargs):
    """Constructs a 16 layers vanilla model."""
    model = Vanilla(*make_layers(16, c, activation), **kwargs)
    return model


def van32(c, activation, **kwargs):
    """Constructs a 32 layers vanilla model."""
    model = Vanilla(*make_layers(32, c, activation), **kwargs)
    return model


def van128(c, activation, **kwargs):
    """Constructs a 128 layers vanilla model."""
    model = Vanilla(*make_layers(128, c, activation), **kwargs)
    return model


def van256(c, activation, **kwargs):
    """Constructs a 256 layers vanilla model."""
    model = Vanilla(*make_layers(256, c, activation), **kwargs)
    return model


def van512(c, activation, **kwargs):
    """Constructs a 512 layers vanilla model."""
    model = Vanilla(*make_layers(512, c, activation), **kwargs)
    return model


def van768(c, activation, **kwargs):
    """Constructs a 768 layers vanilla model."""
    model = Vanilla(*make_layers(768, c, activation), **kwargs)
    return model


def van1024(c, activation, **kwargs):
    """Constructs a 1024 layers vanilla model."""
    model = Vanilla(*make_layers(1024, c, activation), **kwargs)
    return model