import torch
import torch.nn as nn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!= -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BachNorm2d")!=-1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        self.activation = nn.ReLU(inplace=True)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

# same Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)  # , padding)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out

class channel_se(torch.nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(channel_se, self).__init__()

        # Attribute assignment
        # avgpooling,The output weight and high =1
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)

        # the first FC reduce the channel to the 1/4
        self.fc1 = torch.nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)

        # RELU to activate
        self.relu = torch.nn.ReLU()

        # the second fc to recover the channel
        self.fc2 = torch.nn.Linear(in_features=in_channel // (ratio), out_features=in_channel, bias=False)

        # sigmoid activate limit the weight between 0 and 1
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):  # the input is the feature
        b, c, n, t = inputs.shape

        # [b,n,c,t]==>[b,c,n,t]
        # inputs = inputs.reshape(b, c, n, t)

        # pooling [b,c,n,t] ==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # [b,c,1,1]==>[b,c]
        x = x.view([b, c])

        # first [b,c]==>[b,c//4]
        x = self.fc1(x)
        x = self.relu(x)

        # second fc [b,c//4]==>[b,c]
        x = self.fc2(x)
        x = self.sigmoid(x)

        # [b,c] ==> [b,c,1,1]
        x = x.view([b, c, 1, 1])

        outputs = x * inputs
        outputs = outputs.reshape(b, c, n, t)

        return outputs

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # nonlineraity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # encoding layers
        self.conv1 = ConvLayer(62, 64, kernel_size=9, stride=1)
        self.in1_e = nn.BatchNorm2d(64, affine=True)

        self.conv2 = ConvLayer(64, 64, kernel_size=3, stride=2)
        self.in2_e = nn.BatchNorm2d(64, affine=True)

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3_e = nn.BatchNorm2d(128, affine=True)

        self.conv4 = ConvLayer(128, 128, kernel_size=3, stride=2)
        self.in4_e = nn.BatchNorm2d(128, affine=True)

        # residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # decoding layers
        self.deconv4 = UpsampleConvLayer(128, 128, kernel_size=3, stride=1, upsample=2)
        self.in4_d = nn.BatchNorm2d(128, affine=True)

        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in3_d = nn.BatchNorm2d(64, affine=True)

        self.deconv2 = UpsampleConvLayer(64, 64, kernel_size=3, stride=1, upsample=2)
        self.in2_d = nn.BatchNorm2d(64, affine=True)

        self.deconv1 = UpsampleConvLayer(64, 62, kernel_size=9, stride=1)
        self.in1_d = nn.BatchNorm2d(62, affine=True)

        self.Attn0 = Self_Attn(128)
        self.Attn1 = Self_Attn(64)
        self.Attn2 = Self_Attn(64)



    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))

        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.Attn1(y)
        y = self.relu(self.in2_d(self.deconv2(y)))
        y = self.Attn2(y)
        y = self.tanh(self.in1_d(self.deconv1(y)))

        return y


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.layer1 = nn.Sequential(
            # 2200， 62， 64， 64
            nn.Conv2d(62, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm2d(64)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm2d(64)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(4 * 64, 64),
            # nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, 2),
            )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        xb = self.layer3(x2)
        # print(x.shape)
        x3 = xb.view(-1, 4*64)
        x4 = self.layer4(x3)
        return x4, xb, x2, x1
