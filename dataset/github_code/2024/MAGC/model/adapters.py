import torch
import torch.nn as nn


from ldm.models.hyperencoder import myResnetBlock, myDownsample



def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")



class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x
        
class Adapter_XL(nn.Module):

    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=2, ksize=3, sk=True, use_conv=True):
        super(Adapter_XL, self).__init__()

        m = 192
        self.cond_encoder =  nn.Sequential( # 3 256 256 -> 192 32 32
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
        )    

        self.channels = channels
        self.nums_rb = nums_rb
        self.first_rb = myResnetBlock(192 +4, channels[0], None, False)
        # self.conv_in = nn.Conv2d(320, channels[0], 3, 2, 1)        
        self.body = []

        for i in range(len(channels) - 1):
            channel_in = channels[i]
            channel_out = channels[i+1]
            self.body.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=2, padding=1))
            self.body.append(myResnetBlock(channel_out, channel_out))
            self.body.append(myResnetBlock(channel_out, channel_out))

        self.body = nn.ModuleList(self.body)

    def forward(self, x, z_noise):
        # x: b 3 256 256
        semantic_feature = self.cond_encoder(x) # b 192 32 32
        ms_features = []
        latent = torch.cat((semantic_feature, z_noise),dim=1)
        latent = self.first_rb(latent) # b 320 32 32
        ms_features.append(latent) # b 320 32 32


        for i in range(len(self.channels)-1):
            for j in range(self.nums_rb+1): # +1是因为有层卷积
                idx = i * self.nums_rb + j
                latent = self.body[idx](latent)
            ms_features.append(latent)

        return ms_features

