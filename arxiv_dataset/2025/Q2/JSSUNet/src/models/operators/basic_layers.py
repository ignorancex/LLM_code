from torch import nn

class ConvRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvRelu, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, features,  kernel_size):
        super(ResBlock, self).__init__()
        layers = list()
        layers.append(ConvRelu(in_channels=features, out_channels=features,kernel_size=kernel_size))
        layers.append(ConvRelu(in_channels=features, out_channels=features, kernel_size=kernel_size))
        layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=kernel_size//2, bias=False))
        self.cnn=nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.cnn(x)+x)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class ResBlockBN(nn.Module):
    def __init__(self, features, kernel_size):
        super(ResBlockBN, self).__init__()
        layers = list()
        layers.append(ConvRelu(in_channels=features, out_channels=features, kernel_size=kernel_size))
        layers.append(nn.BatchNorm2d(num_features=features))
        layers.append(ConvRelu(in_channels=features, out_channels=features, kernel_size=kernel_size))
        layers.append(nn.BatchNorm2d(num_features=features))
        layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size,
                    padding=kernel_size // 2, bias=False))

        self.cnn = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.cnn(x) + x)

class RCAB(nn.Module):
    def __init__(self, channels, kernel_size, reduction, res_scale=1, **kwargs):
        super(RCAB, self).__init__()

        blocks = [nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size // 2), bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size // 2), bias=True),
                  CALayer(channels, reduction)
                  ]
        self.module = nn.Sequential(*blocks)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.module(x)
        res += x
        return res