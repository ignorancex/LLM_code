import torch.nn as nn
import torch


############################################################
#  base function
############################################################
"default conv layer"
def default_conv(in_channels, out_channels, kernel_size, bias = True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


############################################################
#  Residual Channel Attention Network (RCAN) Generator
############################################################
"Channel Attention (CA) Layer"
class CALayer(nn.Module):
    """
    Channel Attention (CA) Layer, is basical block in RCAN. One CA forms one RCAB(Residual Channel Attention Block).
    See figure 3 of original RCAN paper.
    Beware the CA Layer used in RCAN is actually same as the channel attention mechanism propsed in SENet(Squeeze-and-Excitation Networks).
    """
    def __init__(self, channel, reduction=16):
        """
        reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
        """
        super(CALayer, self).__init__()
        # global average pooling(GAP): feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # global average pooling(GAP), output size is 1 for each channel
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, (1, 1), padding=(0, 0), bias=True), # W_d in CA
                nn.ReLU(inplace=False), # ReLU in CA
                nn.Conv2d(channel // reduction, channel, (1, 1), padding=(0, 0), bias=True), # W_u in CA
                nn.Sigmoid() # sigmoid in CA
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        # the x is the "feature maps over channels" in size C x H x W. The y now is actual the weights in size C x 1 x 1 which represents "channel statistics",
        # it stands for how much "attention" expected to pay for each channel's feature map
        return x * y

"Residual Channel Attention Block (RCAB)"
class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB): There are several RCABs belong to one RG(Residual Group).
    See figure 4 of original RCAN paper
    """
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2): # conv --> ReLU --> conv
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(nn.ReLU(True))
        modules_body.append(CALayer(n_feat, reduction)) # use CA Layer

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x) # conv --> ReLU --> conv --> CA/channel and spatial attention block
        #res = self.body(x).mul(self.res_scale)
        x = x + res # local skip link of RCAB
        return x

"Residual Group (RG)"
class ResidualGroup(nn.Module):
    """
    RG(Residual Group): There are several RGs belong to one RIR(Residual in Residual module).
    See upper figure in figure 2 of original RCAN paper
    """
    def __init__(self, conv, n_feat, kernel_size, reduction, res_scale, n_rcablocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, res_scale=res_scale) \
            for _ in range(n_rcablocks)]

        modules_body.append(conv(n_feat, n_feat, kernel_size)) # last conv after several RCABs as show in figure 2
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x) # RCAB_1 --> RCAB_2 --> ... --> RCAB_(n_rcablocks) --> conv
        x = x + res # short skip connection in RG
        return x

"Residual Channel Attention Network (RCAN)"
class generator_RCAN(nn.Module):
    """
    RCAN(Deep Residual Channel Attention Network) = RIR(Residual in Residual module) + Upsampler Module.
    See bottom figure in figure 2 of original RCAN paper
    """

    def __init__(self, num_in_ch=1, n_resgroups=5, n_rcablocks=5, num_out_ch=1, num_feat=64, scale=1):
        super(generator_RCAN, self).__init__()
        conv = default_conv

        kernel_size = 3  # conv filter size used for all conv in RCAN
        reduction = 16  # reduction is the r mentioned in 3.3 Channel Attention in RCAN paper

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std) """
        # ----------------------------------------------------------------------------------------------------------------------- #

        # define commonly use head module
        modules_head = [
            conv(num_in_ch, num_feat, kernel_size)]  # the first conv layer in RCAN, show in figure 2 of RCAN paper

        # define commonly use pretail module
        modules_pretail = [conv(num_feat, num_feat, kernel_size)]

        # define body module for different type of network.
        # The RIR(Residual in Residual) which consists of n_resgroups RGs, show in figure 2 of RCAN paper
        modules_body = [
            ResidualGroup(
                conv, num_feat, kernel_size, reduction, res_scale=scale, n_rcablocks=n_rcablocks) \
            for _ in range(n_resgroups)]

        # define tail module. The last stage is upsampling module and one more conv layer, show in figure 2 of RCAN paper
        modules_tail = []
        """
        modules_tail.append(
                    Upsampler(scale, num_feat))
        """
        modules_tail.append(conv(num_feat, num_out_ch, kernel_size))
        # modules_tail.append(nn.ReLU(inplace=True))
        modules_tail.append(nn.Tanh())
        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1) """
        # ----------------------------------------------------------------------------------------------------------------------- #

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.pretail = nn.Sequential(*modules_pretail)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # do NOT understand why need this, may NOT be useful for us
        """ x = self.sub_mean(x) """
        x = self.head(x)  # input image goes through first conv layer
        res = self.body(x)  # data goes through sveral ResidualGroups
        res = self.pretail(res)
        x = x + res  # long skip connection of RIR
        #        if (Maintain_Same_Size == True):
        #            res = self.down_size_converter(res) # extra down_size_converter is needed to shtik size of image scale times if we expect same size as input LR for SR output
        x = self.tail(x)  # data goes through upsampling module and one more conv layer
        # do NOT understand why need this, may NOT be useful for us
        """ x = self.add_mean(x) """
        return x


############################################################
#  VGG Discriminator
############################################################
class DiscriminatorVGG(nn.Module):
    def __init__(self, in_ch=3, d=64):
        super(DiscriminatorVGG, self).__init__()
        self.d = d

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # input is 3 x 128 x 128
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.Conv2d(d, d, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),  # state size. 64 x 64 x 64
            # nn.BatchNorm2d(d),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d, d * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.Conv2d(d * 2, d * 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),  # state size. 128 x 32 x 32
            # nn.BatchNorm2d(d * 2),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 2, d * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.Conv2d(d * 4, d * 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),  # state size. 256 x 16 x 16
            # nn.BatchNorm2d(d * 4),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 4, d * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.Conv2d(d * 8, d * 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),  # state size. 512 x 8 x 8
            # nn.BatchNorm2d(d * 8),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 8, d * 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.Conv2d(d * 8, d * 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),  # state size. 512 x 4 x 4
            # nn.BatchNorm2d(d * 8),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(self.d * 8, 100),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Linear(100, 1)
        # )
        self.classifier = nn.Linear(self.d * 16, 1)

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class DiscriminatorVGG1(nn.Module):
    def __init__(self, in_ch=3, d=64):
        super(DiscriminatorVGG1, self).__init__()
        self.d = d

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # input is 3 x 128 x 128
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d, d, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),  # state size. 64 x 64 x 64
            nn.BatchNorm2d(d),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d, d * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 2, d * 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),  # state size. 128 x 32 x 32
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 2, d * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 4, d * 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),  # state size. 256 x 16 x 16
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 4, d * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 8, d * 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),  # state size. 512 x 8 x 8
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 8, d * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 8, d * 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),  # state size. 512 x 4 x 4
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.d * 8, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class DiscriminatorVGG2(nn.Module):
    def __init__(self, in_ch=3, image_size=128, d=64):
        super(DiscriminatorVGG2, self).__init__()
        self.feature_map_size = image_size // 32
        self.d = d

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d, d, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d, d * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 2, d * 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 2, d * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 4, d * 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 4, d * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 8, d * 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 8, d * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 8, d * 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear((self.d * 8) * self.feature_map_size * self.feature_map_size, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out