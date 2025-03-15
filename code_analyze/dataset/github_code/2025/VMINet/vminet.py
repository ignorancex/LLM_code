
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model




class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0., size=56):
        super().__init__()
        self.conv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.sum = nn.Linear(size * size, 1, bias=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=False)
        self.stack = nn.Parameter(torch.ones(2))
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        B,C,H,W = x.size()
        tril = torch.tril(x.reshape(B,C,H*W).permute(0, 2, 1))  #B,H*W,C
        tril = tril.permute(0, 2, 1) #B,C,H*W
        s = self.sum(tril).unsqueeze(2) #B,C,1,1
        x = self.stack[0]*s.expand(B,C,H,W) + self.stack[1]*x
        x = self.g(self.act(x))
        x = input + self.drop_path(x)
        return x


class VMINet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], size=[56, 28, 14, 7], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i], size[i_layer]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)



@register_model
def vminet_Ti(pretrained=False, **kwargs):
    model = VMINet(24, [2, 2, 18, 2], mlp_ratio=2, **kwargs)
    return model


@register_model
def vminet_XS(pretrained=False, **kwargs):
    model = VMINet(48, [2, 2, 18, 2], mlp_ratio=2, **kwargs)
    return model

@register_model
def vminet_S(pretrained=False, **kwargs):
    model = VMINet(48, [2, 2, 18, 2], mlp_ratio=4, **kwargs)
    return model

@register_model
def vminet_B(pretrained=False, **kwargs):
    model = VMINet(96, [2, 2,18, 2], mlp_ratio=2, **kwargs)
    return model



