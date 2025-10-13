import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .refine import Refine
import torchvision.transforms as transforms
from einops import rearrange

T_MAX = 1024 * 1024
# import transforms
from torch.utils.cpp_extension import load

wkv_cuda = load(name="wkv", sources=["./model/cuda/wkv_op.cpp", "./model/cuda/wkv_cuda.cu"],
                verbose=True,
                extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3',
                                   f'-DTmax={T_MAX}'])

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.process(x)
        y = self.avg_pool(res)
        z = self.conv_du(y)
        return z *res + x
class CALayer(nn.Module):
    def __init__(self, channel, reduction):
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
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x
        # return z*x

class Refine(nn.Module):

    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
            # CALayer(n_feat,4),
            # CALayer(n_feat,4),
            ChannelAttention(n_feat, 4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim,
                                         bias=False)
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        # import pdb
        # pdb.set_trace()

        out = self.alpha[0] * x + self.alpha[1] * out1x1 + self.alpha[2] * out3x3 + self.alpha[3] * out5x5
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution

        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))

        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))

        combined_weight = self.alpha[0] * identity_weight + self.alpha[1] * padded_weight_1x1 + self.alpha[
            2] * padded_weight_3x3 + self.alpha[3] * self.conv5x5.weight

        device = self.conv5x5_reparam.weight.device

        combined_weight = combined_weight.to(device)

        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)

    def forward(self, x):

        if self.training:
            self.repram_flag = True
            out = self.forward_train(x)
        elif self.training == False and self.repram_flag == True:
            self.reparam_5x5()
            self.repram_flag = False
            out = self.conv5x5_reparam(x)
        elif self.training == False and self.repram_flag == False:
            out = self.conv5x5_reparam(x)

        return out


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd

        self.dwconv = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)

        self.recurrence = 2

        self.omni_shift = OmniShift(dim=n_embd)

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))
            self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))

    def jit_func(self, x, resolution):
        # Mix x with the previous timestep to produce xk, xv, xr

        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device
        # print(x.device)
        sr, k, v = self.jit_func(x, resolution)

        for j in range(self.recurrence):
            if j % 2 == 0:
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
            else:
                h, w = resolution
                k = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w)
                v = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w)
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
                k = rearrange(k, 'b (w h) c -> b (h w) c', h=h, w=w)
                v = rearrange(v, 'b (w h) c -> b (h w) c', h=h, w=w)

        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd

        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)

        self.omni_shift = OmniShift(dim=n_embd)

        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, resolution):

        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv

        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4,
                 init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, init_mode,
                                    key_norm=key_norm)

        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, hidden_rate,
                                    init_mode, key_norm=key_norm)

        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

    def forward(self, x):
        # x = self.conv1(torch.concat([ms, pan], dim=1))
        b, c, h, w = x.shape

        resolution = (h, w)

        # x = self.dwconv1(x) + x
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.gamma1 * self.att(self.ln1(x), resolution)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        # print(x.shape)
        # x = self.dwconv2(x) + x
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        # input = torch.cat([x,resi],dim=1)
        # out = self.conv_3(input)
        return x + resi


class ConvFuse(nn.Module):
    def __init__(self, in_size, out_size):
        super(ConvFuse, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.out = HinResBlock(out_size, out_size)

    def forward(self, ms, pan):
        out = self.conv1(torch.concat([ms, pan], dim=1))
        return out + self.out(out)

"""
class Causal_Norm_SR(nn.Module):
    def __init__(self, in_channels=32, feat_dim=32, use_effect=True, num_head=2, tau=16.0, alpha=3.0, gamma=0.03125, *args):
        super(Causal_Norm_SR, self).__init__()

        # 参数初始化
        self.in_channels = in_channels  # 输入通道数
        self.feat_dim = feat_dim  # 特征维度
        self.use_effect = use_effect  # 是否使用效果因子
        self.num_head = num_head  # 头数
        self.scale = tau / num_head  # 缩放因子 tau 除以 num_head
        self.norm_scale = gamma  # 归一化缩放因子 gamma
        self.alpha = alpha  # 调整因子 alpha
        self.head_dim = feat_dim // num_head  # 每个头的特征维度

        # 初始化权重矩阵，形状为 (feat_dim, in_channels)，并将其移动到 GPU
        self.weight = nn.Parameter(torch.Tensor(in_channels, feat_dim).cuda(), requires_grad=True)
        self.reset_parameters(self.weight)

        # ReLU 激活函数
        self.relu = nn.ReLU(inplace=True)

        # 添加卷积层以确保输出维度与输入维度一致
        self.conv = nn.Conv2d(feat_dim, in_channels, kernel_size=1)

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def forward(self, x, label=None, embed=None, dif=None):

        # 特征维度
        batch_size, in_channels, height, width = x.size()  # (4, 32, 128, 128)
        if self.training and dif != None:
            return x * dif
        # reshape为 (batch_size, in_channels * height * width)
        x = x.view(batch_size, in_channels * height * width)  # (4, 524288)

        # 多头归一化
        # normed_w 形状为 (32, 524288)
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)  # (32, 4)
        # print("normed_w:",normed_w.shape)
        # 多头 L2 归一化
        # normed_x 形状为 (batch_size, in_channels * height * width)
        normed_x = self.multi_head_call(self.l2_norm, x)  # (4, 524288)
        # print("normed_x:",normed_x.shape)
        # y (batch_size, feat_dim)
        y = torch.mm(normed_x * self.scale, normed_w.t())  # (4, 32)
        # print(y.shape)
        # print(y)
        if (not self.training) and self.use_effect:
            # 转换为张量
            self.embed = torch.from_numpy(embed).view(1, -1).to(x.device)
            # print(self.embed.shape)
            # 多头 L2 归一化
            normed_c = self.multi_head_call(self.l2_norm, self.embed)
            print("normed_c:",normed_c.shape)
            # 将输入特征、混杂因子和权重按照每个头的特征维度进行分割
            x_list = torch.split(normed_x, self.head_dim, dim=1)
            c_list = torch.split(normed_c, self.head_dim, dim=1)
            w_list = torch.split(normed_w, self.head_dim, dim=1)
            output = []
            for nx, nc, nw in zip(x_list, c_list, w_list):
                cos_val, sin_val = self.get_cos_sin(nx, nc)
                y0 = torch.mm((nx - cos_val * self.alpha * nc) * self.scale, nw.t())
                output.append(y0)
            # 将所有头的输出求和
            y = sum(output) # (4, 32)
        y = torch.sigmoid(y)
        # print(y)
        # print(y.view(batch_size, self.feat_dim, height, width))
        # 使用卷积层调整输出维度，使其与输入维度匹配
        # y = self.conv(y.view(batch_size, self.feat_dim, height, width))  # (4, 32, 128, 128)
        y = y.view(batch_size, in_channels, 1, 1)
        return y

    def get_cos_sin(self, x, y):
        # 计算 x 和 y 之间的余弦值和正弦值
        cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

    def multi_head_call(self, func, x, weight=None):
        # 将输入 x 按照每个头的特征维度进行分割
        x_list = torch.split(x, self.head_dim, dim=1)
        if weight:
            # 如果提供了权重，则对每个分割后的部分应用函数 func 和 weight
            y_list = [func(item, weight) for item in x_list]
        else:
            # 如果没有提供权重，则对每个分割后的部分应用函数 func
            y_list = [func(item) for item in x_list]
        # 确保分割后的部分数目与头数一致
        assert len(x_list) == self.num_head
        assert len(y_list) == self.num_head
        # 将分割后的部分重新拼接
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        # 对 x 进行 L2 归一化
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def causal_norm(self, x, weight):
        # 对 x 进行因果归一化
        norm = torch.norm(x, 2, 1, keepdim=True)
        normed_x = x / (norm + weight)
        return normed_x
"""

class Causal_Norm_SR(nn.Module):
    def __init__(self, in_channels=32, feat_dim=32, use_effect=True, num_head=2, tau=16.0, alpha=3.0, gamma=0.03125, *args):
        super(Causal_Norm_SR, self).__init__()

        # 参数初始化
        self.in_channels = in_channels  # 输入通道数
        # feat_dim = 1049600
        # self.feat_dim = 1049600  # 特征维度
        self.feat_dim = feat_dim
        self.use_effect = use_effect  # 是否使用效果因子
        self.num_head = num_head  # 头数
        self.scale = tau / num_head  # 缩放因子 tau 除以 num_head
        self.norm_scale = gamma  # 归一化缩放因子 gamma
        self.alpha = alpha  # 调整因子 alpha
        self.head_dim = self.feat_dim // num_head  # 每个头的特征维度
        # print(self.head_dim) # 16384
        # 初始化权重矩阵，形状为 (feat_dim, in_channels)，并将其移动到 GPU
        self.weight = nn.Parameter(torch.Tensor(in_channels, feat_dim).cuda(), requires_grad=True)
        self.reset_parameters(self.weight)

        # ReLU 激活函数
        self.relu = nn.ReLU(inplace=True)

        # 添加卷积层以确保输出维度与输入维度一致
        self.conv = nn.Conv2d(feat_dim, in_channels, kernel_size=1)

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def forward(self, x, label=None, embed=None, dif=None):
        # 获取输入张量的形状
        batch_size, in_channels, height, width = x.size()  # (4, 32, 128, 128)
        # print(batch_size, in_channels, height, width) # (1, 32, 1024, 1024)
        # 确保 dif 的维度与 x 匹配
        if self.training and dif is not None:
            return x * dif
        if dif is not None:
            dif = dif.expand_as(x)  # 将 dif 扩展到 (4, 32, 128, 128) (1, 32, 1024, 1024)
        # 如果处于训练模式且存在 dif，则返回 x * dif


        # 将 x 和 dif 降维为 (batch_size, in_channels, height * width)
        x_flat = x.view(batch_size, in_channels, -1)  # (4, 32, 128 * 128) (1, 32, 1024*1024)

        if dif is not None:
            dif_flat = dif.view(batch_size, in_channels, -1)  # (4, 32, 128 * 128) (1, 32, 1024*1024) (1, 32, 512*512)
        # 使用 dif 代替权重进行运算
        # 多头归一化
        normed_w = self.multi_head_call(self.causal_norm, dif_flat, input_dim=2, weight=self.norm_scale)   # (4, 32, 128 * 128)
        # print("normed_w.shape:", normed_w.shape)
        # 多头归一化
        # normed_w = dif_flat
        normed_x = self.multi_head_call(self.l2_norm, x_flat, input_dim=2)  # (4, 32, 128 * 128)
        # print("normed_x.shape:", normed_x.shape)
        # 计算 y
        y = normed_x * self.scale * normed_w  # 逐元素相乘 (4, 32, 128 * 128)
        # y = normed_x * normed_w
        # print(y)
        if not self.training and self.use_effect:
            # 将 embed 转换为张量
            self.embed = torch.from_numpy(embed).view(1, -1).to(x.device)
            # print("embed.shape:", self.embed.shape)
            # 将 embed 升维为 (1, in_channels, height * width)

            # normed_c = self.multi_head_call(self.l2_norm, self.embed,input_dim=1)
            t_list = torch.split(x, 524288, dim=1)
            # print("t_list[0].shape:", t_list[0].shape)
            y_list = [self.l2_norm(item) for item in t_list]

            normed_c = torch.cat(y_list, dim=1)
            normed_c = normed_c.view(1, in_channels, -1).expand(batch_size, in_channels, -1)

            # 将输入特征、混杂因子和权重按照每个头的特征维度进行分割
            x_list = torch.split(normed_x, self.head_dim, dim=2)
            c_list = torch.split(normed_c, self.head_dim, dim=2)
            w_list = torch.split(normed_w, self.head_dim, dim=2)
            output = []
            for nx, nc, nw in zip(x_list, c_list, w_list):
                cos_val, sin_val = self.get_cos_sin(nx, nc)
                y0 = (nx - cos_val * self.alpha * nc) * self.scale * nw
                # print("y0.shape:", y0.shape)
                output.append(y0)
            # 将所有头的输出求和
            y = sum(output)  # (4, 32, 128 * 128)
            y = torch.sigmoid(y)
            # print(y)
        # print("y.shape:", y.shape)
        # 将 y 重新变换为 (4, 32, 128, 128)

        y = y.view(batch_size, in_channels, height, width)
        return y

    def get_cos_sin(self, x, y):
        # 计算 x 和 y 之间的余弦值和正弦值
        cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

    def multi_head_call(self, func, x, input_dim=None, weight=None):
        # 将输入 x 按照每个头的特征维度进行分割
        x_list = torch.split(x, self.head_dim, dim=input_dim)
        if weight:
            # 如果提供了权重，则对每个分割后的部分应用函数 func 和 weight
            y_list = [func(item, weight) for item in x_list]
        else:
            # 如果没有提供权重，则对每个分割后的部分应用函数 func
            y_list = [func(item) for item in x_list]
        # 确保分割后的部分数目与头数一致
        # print("len x_List:", len(x_list))
        # print("x_list:", x_list[0].shape)
        assert len(x_list) == self.num_head
        assert len(y_list) == self.num_head
        # 将分割后的部分重新拼接
        return torch.cat(y_list, dim=input_dim)

    def l2_norm(self, x):
        # 对 x 进行 L2 归一化
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def causal_norm(self, x, weight):
        # 对 x 进行因果归一化
        norm = torch.norm(x, 2, 1, keepdim=True)
        normed_x = x / (norm + weight)
        return normed_x


class Net(nn.Module):
    def __init__(self, num_channels=None, base_filter=None, args=None):
        super(Net, self).__init__()
        base_filter = 32
        self.feat_dim = 524288 # channel * H * W
        # 8388608  #  33554432
        self.base_filter = base_filter
        self.fliper = transforms.RandomHorizontalFlip(1)
        self.pan_encoder = nn.Sequential(nn.Conv2d(1, base_filter, 3, 1, 1), HinResBlock(base_filter, base_filter),
                                         HinResBlock(base_filter, base_filter), HinResBlock(base_filter, base_filter))
        self.ms_encoder = nn.Sequential(nn.Conv2d(4, base_filter, 3, 1, 1), HinResBlock(base_filter, base_filter),
                                        HinResBlock(base_filter, base_filter), HinResBlock(base_filter, base_filter))

        self.deep_fusion1 = ConvFuse(base_filter * 2, base_filter)
        self.deep_fusion2 = ConvFuse(base_filter * 2, base_filter)
        self.deep_fusion3 = ConvFuse(base_filter * 2, base_filter)
        self.deep_fusion4 = ConvFuse(base_filter * 2, base_filter)
        self.deep_fusion5 = ConvFuse(base_filter * 2, base_filter)

        num_blocks = [1, 2, 2, 2, 3]
        # self.encoder_level1 = nn.Sequential(
            # *[Block(n_embd=base_filter, n_layer=num_blocks[0], layer_id=i) for i in range(num_blocks[0])])
        # self.down1_2 = Downsample(base_filter)  ## From Level 1 to Level 2
        # self.encoder_level2 = nn.Sequential(
            # *[Block(n_embd=base_filter, n_layer=num_blocks[1], layer_id=i) for i in range(num_blocks[1])])
        # self.down2_3 = Downsample(int(base_filter * 2 ** 1))  ## From Level 2 to Level 3

        """
        self.latent = nn.Sequential(
            *[Block(n_embd=int(base_filter * 2 ** 2), n_layer=num_blocks[2], layer_id=i) for i in range(num_blocks[2])])
        self.up3_2 = Upsample(int(base_filter * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(base_filter * 2 ** 2), int(base_filter * 2 ** 1), kernel_size=1, bias=True)
        self.decoder_level2 = nn.Sequential(
            *[Block(n_embd=int(base_filter* 2 ** 1), n_layer=num_blocks[1], layer_id=i) for i in range(num_blocks[1])])
        self.up2_1 = Upsample(int(base_filter * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(
            *[Block(n_embd=int(base_filter), n_layer=num_blocks[0], layer_id=i) for i in range(num_blocks[0])])
        self.reduce_chan_level1 = nn.Conv2d(int(base_filter * 2 ** 1), int(base_filter), kernel_size=1, bias=True)
        """

        # self.encoder_level3 = nn.Sequential(
            # *[Block(n_embd=base_filter, n_layer=num_blocks[2], layer_id=i) for i in range(num_blocks[2])])
        # self.encoder_level4 = nn.Sequential(
            # *[Block(n_embd=base_filter, n_layer=num_blocks[3], layer_id=i) for i in range(num_blocks[3])])
        # self.encoder_level5 = nn.Sequential(
            # *[Block(n_embd=base_filter, n_layer=num_blocks[4], layer_id=i) for i in range(num_blocks[4])])
        # self.encoder_level6 = nn.Sequential(
            # *[Block(n_embd=base_filter, n_layer=num_blocks[5], layer_id=i) for i in range(num_blocks[4])])
        # self.conv_pre = nn.Conv2d(base_filter * 2, base_filter, kernel_size=3, padding=1)
        # self.conv_pre2 = nn.Conv2d(base_filter * 2, base_filter, kernel_size=3, padding=1)
        # self.conv_pre3 = nn.Conv2d(base_filter * 2, base_filter, kernel_size=3, padding=1)
        # self.conv_pre4 = nn.Conv2d(base_filter * 2, base_filter, kernel_size=3, padding=1)
        # self.conv_pre5 = nn.Conv2d(base_filter * 2, base_filter, kernel_size=3, padding=1)
        # self.suft = SUFT(dp_feats=32, add_feats=32, scale=4)
        # self.deep_fusion6 = ConvFuse(base_filter * 2, base_filter)
        # self.deep_fusion7 = ConvFuse(base_filter * 2, base_filter)
        self.pan_feature_extraction = nn.Sequential(*[HinResBlock(base_filter, base_filter) for i in range(8)])
        self.ms_feature_extraction = nn.Sequential(*[HinResBlock(base_filter, base_filter) for i in range(8)])
        self.embed_mean = torch.zeros(int(self.feat_dim)).numpy()
        self.causal = Causal_Norm_SR(in_channels=32, feat_dim=16384, use_effect=True, num_head=1, tau=16.0, alpha=3.0, gamma=0.03125)
        # feat_dim match self.feat_dim
        self.output = Refine(base_filter, 4)

        # self.conv_dp1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # self.dp_rg1 = ResidualGroup(default_conv, n_feat=32, kernel_size=3, reduction=16, n_resblocks=4)
        # self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.conv_su = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=4, padding=1)
        self.conv_du = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)

    def calculate_uncertainty(self, ms_f, ms_flip_f):
        dif = torch.abs(ms_f - self.fliper(ms_flip_f))

        dif_avg = torch.mean(dif, dim=1, keepdim=True)
        dif_max, _ = torch.max(dif, dim=1, keepdim=True)
        attention = self.conv_du(torch.cat([dif_avg, dif_max], dim=1))

        max = torch.max(torch.max(attention, -1)[0], -1)[0].unsqueeze(1).unsqueeze(2)
        min = torch.min(torch.min(attention, -1)[0], -1)[0].unsqueeze(1).unsqueeze(2)

        attention = (attention - min) / (max - min + 1e-12)
        return attention


    def forward(self, ms, _, pan, mu=0.9):
        # print(ms.shape)
        # raw_ms = ms
        ms_flip = self.fliper(ms)
        pan_flip = self.fliper(pan)

        ms_bic = F.interpolate(ms, scale_factor=4)
        ms_bic_flip = F.interpolate(ms_flip, scale_factor=4)

        ms_f = self.ms_encoder(ms_bic)


        ms_flip_f = self.ms_encoder(ms_bic_flip)
        b, c, h, w = ms_f.shape # 4 * 32 * 128 * 128
        pan_f = self.pan_encoder(pan)
        pan_flip_f = self.pan_encoder(pan_flip)
        ms_f = self.ms_feature_extraction(ms_f) # 4 * 32 * 128 * 128
        # print(ms_f.shape) 1, 32, 512, 512
        pan_f = self.pan_feature_extraction(pan_f)
        ms_flip_f = self.ms_feature_extraction(ms_flip_f)
        pan_flip_f = self.pan_feature_extraction(pan_flip_f)

        ms_f = self.deep_fusion1(ms_f, pan_f)
        ms_f = self.deep_fusion2(ms_f, pan_f)
        ms_f = self.deep_fusion3(ms_f, pan_f)
        ms_f = self.deep_fusion4(ms_f, pan_f)
        ms_f = self.deep_fusion5(ms_f, pan_f)  # 4 * 32 * 128 * 128

        # 2 3 3 3 4      1 2 2 2 3     1 2 2 3

        # ms_f = self.conv_pre(torch.concat([ms_f, pan_f], dim=1)) # 4 * 32 * 128 * 128

        # ms_f = self.encoder_level1(ms_f)

        # ms_f = self.conv_pre2(torch.concat([ms_f, pan_f], dim=1))
        # ms_f = self.encoder_level2(ms_f)
        # ms_f = self.conv_pre3(torch.concat([ms_f, pan_f], dim=1))
        # ms_f = self.encoder_level3(ms_f)
        # ms_f = self.conv_pre4(torch.concat([ms_f, pan_f], dim=1))
        # ms_f = self.encoder_level4(ms_f)
        # ms_f = self.conv_pre5(torch.concat([ms_f, pan_f], dim=1))
        # ms_f = self.encoder_level5(ms_f)


        # print(ms_f.shape)

        ms_flip_f = self.deep_fusion1(ms_flip_f, pan_flip_f)
        ms_flip_f = self.deep_fusion2(ms_flip_f, pan_flip_f)
        ms_flip_f = self.deep_fusion3(ms_flip_f, pan_flip_f)
        ms_flip_f = self.deep_fusion4(ms_flip_f, pan_flip_f)
        ms_flip_f = self.deep_fusion5(ms_flip_f, pan_flip_f)


        # ms_f = self.deep_fusion6(ms_f, pan_f)
        # ms_f = self.deep_fusion7(ms_f, pan_f)
        # ms_de = self.output(ms_de)
        # print(ms_f.shape)
        # print(self.output(ms_f).shape)

        # dif = torch.abs(ms_f - self.fliper(ms_flip_f))
        attention = self.calculate_uncertainty(ms_f, ms_flip_f)
        # print(ms_f.shape)
        self.embed_mean = mu * self.embed_mean + ms_f.detach().mean(0).view(-1).cpu().numpy()
        embed_mean = self.embed_mean
        de_f = self.causal(ms_f, embed=self.embed_mean, dif=attention)
        # print(de_f)

        if self.training:
            ms_f = 0.9 * ms_f + 0.1 * de_f
        else:
            ms_f = 0.9 * ms_f + 0.1 * ms_f * de_f
        # feature = ms_f
        hrms = self.output(ms_f) + ms_bic
        # print(hrms.shape)
        return hrms

if __name__ == "__main__":
    from torchvision.transforms import Compose, ToTensor
    def transform():
        return Compose([
            ToTensor(),
        ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = Net().to(device)  
    model.eval()
    
    # Test with 128x128 input
    ms = torch.ones((1, 4, 32, 32)).to(device)  # LR MS image
    bms = torch.ones((1, 4, 128, 128)).to(device)  # Bicubic upsampled MS
    pan = torch.ones((1, 1, 128, 128)).to(device)  # PAN image
    mu = 0.9  # momentum parameter
    
    output = model(ms, bms, pan, mu)
    print("Output shape:", output.shape)
