import torch
import torch.nn as nn
from thop import profile
from einops import rearrange


class MHSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MHSA, self).__init__()
        self.num_heads = num_heads
        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) / (c // self.num_heads) ** 0.5
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class AB(nn.Module):
    def __init__(self, dim, heads, bias):
        super(AB, self).__init__()
        self.attn = MHSA(dim=dim, num_heads=heads, bias=bias)
        self.FFN = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 4, dim, kernel_size=1)
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.FFN(x)
        return x


class IEB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IEB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.match_channels = None
        if in_channels != out_channels:
            self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.match_channels:
            residual = self.match_channels(residual)
        out += residual
        out = self.relu(out)
        return out



class UNIRNet(nn.Module):
    def __init__(self):
        super(UNIRNet, self).__init__()


        self.initial_conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.Sequential(
            IEB(32, 64),
            AB(64, heads=4, bias=False),
            IEB(64, 64)
        )


        self.middle_blocks = nn.Sequential(
            AB(64, heads=4, bias=False),
            IEB(64, 64)
        )


        self.decoder = nn.Sequential(
            IEB(64, 32),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        )

        self.VRM = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):

        x = self.initial_conv(x)


        x = self.encoder(x)


        x = self.middle_blocks(x)


        x = self.decoder(x)

        x = self.VRM(x)

        return x


