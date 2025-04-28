import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from einops import rearrange


def A_operator(z, Phi):
    y = torch.sum(Phi * z, 1, keepdim=True)
    return y


def At_operator(z, Phi):
    y = z * Phi
    return y


## Adapt from "Restormer: Efficient Transformer for High-Resolution Image Restoration"
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b t h w c')
        x = self.body(x)
        x = rearrange(x, 'b t h w c -> b c t h w')
        return x.contiguous()


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv3d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv3d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.gelu = nn.GELU()
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv1 = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.Conv3d(dim, dim*3, 1, 1, 0)
        )
        self.qkv2 = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.Conv3d(dim, dim*3, 1, 1, 0)
        )

        self.project1 = nn.Conv3d(dim, dim, 1, 1, 0)
        self.project2 = nn.Conv3d(dim, dim, 1, 1, 0)

    def forward(self, x1, x2):
        b, c, t, h, w = x1.shape

        q1, k1, v1 = self.qkv1(x1).chunk(3, dim=1)
        q2, k2, v2 = self.qkv2(x2).chunk(3, dim=1)
        q1 = rearrange(q1, 'b (head c) t h w -> b head c (t h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) t h w -> b head c (t h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) t h w -> b head c (t h w)', head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn = (q1 @ k2.transpose(-2, -1)) * self.temperature1
        attn = attn.softmax(dim=-1)
        out1 = (attn @ v2)
        out1 = rearrange(out1, 'b head c (t h w) -> b (head c) t h w', head=self.num_heads, t=t, h=h, w=w)

        q2 = rearrange(q2, 'b (head c) t h w -> b head c (t h w)', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) t h w -> b head c (t h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) t h w -> b head c (t h w)', head=self.num_heads)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        attn = (q2 @ k1.transpose(-2, -1)) * self.temperature2
        attn = attn.softmax(dim=-1)
        out2 = (attn @ v1)
        out2 = rearrange(out2, 'b head c (t h w) -> b (head c) t h w', head=self.num_heads, t=t, h=h, w=w)

        x1 = self.project1(out1)
        x2 = self.project2(out2)
        return x1, x2


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.attn = Attention(dim, num_heads)

        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x1, x2):
        x1_, x2_ = self.attn(self.norm1(x1), self.norm2(x2))
        x1, x2 = x1 + x1_, x2 + x2_
        x1 = x1 + self.ffn1(x1)
        x2 = x2 + self.ffn2(x2)
        return x1, x2


class Gated3DConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Gated3DConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, 1, 1),
        )
        self.gate = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(x)*self.conv(x) + x


class LinearProj(torch.nn.Module):
    def __init__(self):
        super(LinearProj, self).__init__()
        self.eta_step = nn.Parameter(torch.Tensor([0.01]))

    def forward(self, v, y, Phi, Phisum):

        yb = A_operator(v, Phi)
        v = v + At_operator(torch.div(y - yb, Phisum + self.eta_step), Phi)

        return v


class OneStage(torch.nn.Module):
    def __init__(self, in_chans, embed_dim, index, use_checkpoint):
        super(OneStage, self).__init__()
        self.linear_proj = LinearProj()

        self.embed = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Gated3DConvBlock(embed_dim, embed_dim)
        )

        self.down1 = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim*2, kernel_size=3, padding=1, stride=(1, 2, 2), bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Gated3DConvBlock(embed_dim*2, embed_dim*2)
        )

        self.down2 = nn.Sequential(
            nn.Conv3d(embed_dim*2, embed_dim*2*2, kernel_size=3, padding=1, stride=(1, 2, 2), bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Gated3DConvBlock(embed_dim*2*2, embed_dim*2*2)
        )

        self.cross_attn = TransformerBlock(
            dim=embed_dim*2*2,
            num_heads=4, ffn_expansion_factor=2, bias=True,
        ) if index != 0 else None

        self.upc2 = nn.Sequential(
            nn.Conv3d(embed_dim*2*2, embed_dim*2, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Gated3DConvBlock(embed_dim*2, embed_dim*2),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
        )
        self.upc1 = nn.Sequential(
            nn.Conv3d(embed_dim*2, embed_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Gated3DConvBlock(embed_dim, embed_dim),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
        )
        self.neck = nn.Sequential(
            Gated3DConvBlock(embed_dim, embed_dim),
            nn.Conv3d(embed_dim, in_chans, kernel_size=3, padding=1, bias=True),
        )

        self.use_checkpoint = use_checkpoint

    def forward(self, v, y, Phi, Phisum, h=None):
        x = self.linear_proj(v, y, Phi, Phisum)
        v = x.transpose(1, 2).contiguous()

        x0 = self.embed(v)
        x1 = self.down1(x0)
        x2 = self.down2(x1)

        if h is None:
            h = x2
        else:
            if self.use_checkpoint:
                x2, h = cp.checkpoint(self.cross_attn, x2, h)
            else:
                x2, h = self.cross_attn(x2, h)

        x2 = self.upc2(x2)
        x1 = self.upc1(x1 + x2)
        x = self.neck(x0 + x1)

        x = v + x
        v = x.transpose(1, 2).contiguous()
        return v, h


class DUN(torch.nn.Module):
    def __init__(self, in_chans=1, embed_dim=32, num_stage=10, use_checkpoint=False):
        super(DUN, self).__init__()
        self.in_chans = in_chans
        self.num_stage = num_stage

        stages = []
        for i in range(num_stage):
            stages.append(OneStage(in_chans, embed_dim, i, use_checkpoint))
        self.stages = nn.ModuleList(stages)

    def forward(self, y, Phi):
        Phisum = torch.sum(Phi * Phi, 1, keepdim=True)
        Phisum[Phisum == 0] = 1.

        v = At_operator(y, Phi)
        h = None

        for stage in self.stages:
            v, h = stage(v, y, Phi, Phisum, h)

        return v, h


class Down(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(dims, dims // 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.PixelUnshuffle(2),
            nn.Conv2d(dims * 2, dims * 2, 3, 1, 1),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(dims, dims * 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(dims // 2, dims // 2, 3, 1, 1),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dims, dims // 2, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dims // 2, dims // 2, 3, 1, 1)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, in_chans):
        super(UNet2D, self).__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
        )
        self.down1 = (Down(32))
        self.down2 = (Down(64))
        self.down3 = (Down(128))
        self.up1 = (Up(256))
        self.up2 = (Up(128))
        self.up3 = (Up(64))
        self.outc = nn.Conv2d(32, in_chans, 3, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x


class BMNet(torch.nn.Module):
    def __init__(self, in_chans=1, num_stage=10, embed_dim=32, cs_ratio=(4, 4), scaler=False, use_checkpoint=False):
        super(BMNet, self).__init__()
        self.dun = DUN(in_chans=in_chans, embed_dim=embed_dim, num_stage=num_stage, use_checkpoint=use_checkpoint)
        self.refinement = UNet2D(in_chans=in_chans)
        self.cr = cs_ratio
        self.scaler = nn.Parameter(torch.Tensor([1.0]), requires_grad=scaler)

    def forward(self, y, Phi, output_hidden=False):
        cr1, cr2 = self.cr

        inital, hidden = self.dun(y, Phi)

        inital = rearrange(inital, "b (cr1 cr2) c h w -> b c (cr1 h) (cr2 w)", cr1=cr1, cr2=cr2)
        final = inital + self.refinement(inital)

        if output_hidden:
            return final, hidden
        else:
            return final