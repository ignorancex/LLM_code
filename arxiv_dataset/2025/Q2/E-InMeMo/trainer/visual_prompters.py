"""
    source1(VP): https://github.com/hjbahng/visual_prompting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import transforms
import numpy as np
import os
from transformers import ViTFeatureExtractor, ResNetModel, ViTModel, ViTMAEModel, ViTForImageClassification


class PadPrompter(nn.Module):
    def __init__(self, p_eps):
        super(PadPrompter, self).__init__()
        self.pad_size = 30
        image_size = 224
        self.p_eps = p_eps

        self.base_size = image_size - self.pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, self.pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, self.pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - self.pad_size*2, self.pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - self.pad_size*2, self.pad_size]))

    def forward(self, x):
        n_samples = x.shape[0]
        base = torch.zeros(1, 3, self.base_size, self.base_size).to(x.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])
        x_prompted = x + self.p_eps * prompt
        return x_prompted


class PaddingPrompter(nn.Module):
    def __init__(self, pad_size, out_image_size, device):
        super(PaddingPrompter, self).__init__()
        self.pad_size = pad_size
        self.out_image_size = out_image_size
        self.device = device

        self.pad_up = nn.Parameter(torch.randn([1, 3, self.pad_size, self.out_image_size], device=self.device))
        self.pad_down = nn.Parameter(torch.randn([1, 3, self.pad_size, self.out_image_size], device=self.device))
        self.pad_left = nn.Parameter(
            torch.randn([1, 3, self.out_image_size - self.pad_size * 2, self.pad_size], device=self.device))
        self.pad_right = nn.Parameter(
            torch.randn([1, 3, self.out_image_size - self.pad_size * 2, self.pad_size], device=self.device))

    def forward(self, x):
        base = torch.zeros(1, 3, self.out_image_size - self.pad_size * 2, self.out_image_size - self.pad_size * 2,
                           device=self.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])
        return prompt


class SRDPadPrompter(nn.Module):
    def __init__(self, p_eps, mode, pad_size):
        super(SRDPadPrompter, self).__init__()
        self.pad_size = pad_size
        full_image_size = 224

        self.p_eps = p_eps
        self.mode = mode

        if self.mode == 'spimg_spmask':
            target_height = 112
            target_width = 224
            self.base_height = target_height - self.pad_size * 2
            self.base_width = target_width
            self.pad_up = nn.Parameter(torch.randn([1, 3, self.pad_size, self.base_width]))
            self.pad_down = nn.Parameter(torch.randn([1, 3, self.pad_size, self.base_width]))
            self.pad_left = nn.Parameter(torch.randn([1, 3, self.base_height, self.pad_size]))
            self.pad_right = nn.Parameter(torch.randn([1, 3, self.base_height, self.pad_size]))

        elif self.mode == 'spimg':
            target_height = 112
            self.base_height = target_height - self.pad_size * 2
            self.base_width = target_height - self.pad_size * 2
            self.pad_up = nn.Parameter(torch.randn([1, 3, self.pad_size, target_height]))
            self.pad_down = nn.Parameter(torch.randn([1, 3, self.pad_size, target_height]))
            self.pad_left = nn.Parameter(torch.randn([1, 3, self.base_height, self.pad_size]))
            self.pad_right = nn.Parameter(torch.randn([1, 3, self.base_height, self.pad_size]))

        elif self.mode == 'spimg_qrimg':
            target_height = 224
            target_width = 112
            self.base_height = target_height - self.pad_size * 2
            self.base_width = target_width
            self.pad_up = nn.Parameter(torch.randn([1, 3, self.pad_size, self.base_width]))
            self.pad_down = nn.Parameter(torch.randn([1, 3, self.pad_size, self.base_width]))
            self.pad_left = nn.Parameter(torch.randn([1, 3, self.base_height, self.pad_size]))
            self.pad_right = nn.Parameter(torch.randn([1, 3, self.base_height, self.pad_size]))

        elif self.mode == 'qrimg':
            target_height = 112
            self.base_height = target_height - self.pad_size * 2
            self.base_width = target_height - self.pad_size * 2
            self.pad_up = nn.Parameter(torch.randn([1, 3, self.pad_size, target_height]))
            self.pad_down = nn.Parameter(torch.randn([1, 3, self.pad_size, target_height]))
            self.pad_left = nn.Parameter(torch.randn([1, 3, self.base_height, self.pad_size]))
            self.pad_right = nn.Parameter(torch.randn([1, 3, self.base_height, self.pad_size]))

        elif self.mode == 'spimg_spmask_qrimg':
            target_height = 112
            self.base_height = target_height - self.pad_size * 2
            self.base_width = target_height - self.pad_size * 2
            self.pad_up1 = nn.Parameter(torch.randn([1, 3, self.pad_size, target_height]))
            self.pad_down1 = nn.Parameter(torch.randn([1, 3, self.pad_size, target_height]))
            self.pad_left1 = nn.Parameter(torch.randn([1, 3, self.base_height, self.pad_size]))
            self.pad_right1 = nn.Parameter(torch.randn([1, 3, self.base_height, self.pad_size]))

            self.pad_up2 = nn.Parameter(torch.randn([1, 3, self.pad_size, target_height]))
            self.pad_down2 = nn.Parameter(torch.randn([1, 3, self.pad_size, target_height]))
            self.pad_left2 = nn.Parameter(torch.randn([1, 3, self.base_height, self.pad_size]))
            self.pad_right2 = nn.Parameter(torch.randn([1, 3, self.base_height, self.pad_size]))

            self.pad_up3 = nn.Parameter(torch.randn([1, 3, self.pad_size, target_height]))
            self.pad_down3 = nn.Parameter(torch.randn([1, 3, self.pad_size, target_height]))
            self.pad_left3 = nn.Parameter(torch.randn([1, 3, self.base_height, self.pad_size]))
            self.pad_right3 = nn.Parameter(torch.randn([1, 3, self.base_height, self.pad_size]))

    def forward(self, x):
        device = x.device
        if self.mode in ['spimg_spmask', 'spimg', 'spimg_qrimg', 'qrimg', 'spimg_spmask_qrimg']:
            if self.mode == 'spimg_spmask':
                base = torch.zeros(1, 3, self.base_height, self.base_width - 2 * self.pad_size, device=device)
                prompt_horizontal = torch.cat([self.pad_left, base, self.pad_right], dim=3)
                prompt = torch.cat([self.pad_up, prompt_horizontal, self.pad_down], dim=2)

                pad_height = 224 - 112
                prompt_full = F.pad(prompt, (0, 0, 0, pad_height), mode='constant', value=0)

            elif self.mode == 'spimg':
                base = torch.zeros(1, 3, self.base_height, self.base_width, device=device)
                prompt_horizontal = torch.cat([self.pad_left, base, self.pad_right], dim=3)
                prompt = torch.cat([self.pad_up, prompt_horizontal, self.pad_down], dim=2)
                pad_height = 224 - 112
                prompt_full = F.pad(prompt, (0, pad_height, 0, pad_height), mode='constant', value=0)

            elif self.mode == 'spimg_qrimg':
                base = torch.zeros(1, 3, self.base_height, self.base_width - 2 * self.pad_size, device=device)
                prompt_horizontal = torch.cat([self.pad_left, base, self.pad_right], dim=3)
                prompt = torch.cat([self.pad_up, prompt_horizontal, self.pad_down], dim=2)

                pad_width = 224 - 112
                prompt_full = F.pad(prompt, (0, pad_width, 0, 0), mode='constant', value=0)

            elif self.mode == 'qrimg':
                base = torch.zeros(1, 3, self.base_height, self.base_width, device=device)
                prompt_horizontal = torch.cat([self.pad_left, base, self.pad_right], dim=3)
                prompt = torch.cat([self.pad_up, prompt_horizontal, self.pad_down], dim=2)
                pad_height = 224 - 112
                prompt_full = F.pad(prompt, (0, pad_height, pad_height, 0), mode='constant', value=0)

            elif self.mode == 'spimg_spmask_qrimg':
                base1 = torch.zeros(1, 3, self.base_height, self.base_width, device=device)
                prompt_horizontal1 = torch.cat([self.pad_left1, base1, self.pad_right1], dim=3)
                prompt1 = torch.cat([self.pad_up1, prompt_horizontal1, self.pad_down1], dim=2)

                base2 = torch.zeros(1, 3, self.base_height, self.base_width, device=device)
                prompt_horizontal2 = torch.cat([self.pad_left2, base2, self.pad_right2], dim=3)
                prompt2 = torch.cat([self.pad_up2, prompt_horizontal2, self.pad_down2], dim=2)

                base3 = torch.zeros(1, 3, self.base_height, self.base_width, device=device)
                prompt_horizontal3 = torch.cat([self.pad_left3, base3, self.pad_right3], dim=3)
                prompt3 = torch.cat([self.pad_up3, prompt_horizontal3, self.pad_down3], dim=2)

                prompt_inter = torch.cat([prompt1, prompt3], dim=2)
                # print('prompt_inter size: ', prompt_inter.shape)  # prompt_inter size:  torch.Size([1, 3, 224, 112])
                pad_height = 224 - 112
                prompt2 = F.pad(prompt2, (0, 0, 0, pad_height), mode='constant', value=0)

                prompt_full = torch.cat([prompt_inter, prompt2], dim=3)

                # print('prompt_full size: ', prompt_full.shape)

            prompt_full = prompt_full.expand(x.size(0), -1, -1, -1)

            assert x.shape == prompt_full.shape
            x_prompted = x + self.p_eps * prompt_full

        else:
            x_prompted = x

        return x_prompted


class AutoPadPrompter(nn.Module):
    def __init__(self, p_eps, scale, set_resize=False):
        super(AutoPadPrompter, self).__init__()
        self.set_resize = set_resize
        out_image_size = 224
        img_size = 128 * scale
        self.pad_size = int((out_image_size - img_size) // 2)
        self.p_eps = p_eps

        self.base_size = out_image_size - self.pad_size*2
        assert self.base_size == img_size
        self.pad_up = nn.Parameter(torch.randn([1, 3, self.pad_size, out_image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, self.pad_size, out_image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, out_image_size - self.pad_size*2, self.pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, out_image_size - self.pad_size*2, self.pad_size]))

    def forward(self, x):
        n_samples = x.shape[0]
        channel = x.shape[1]
        x = x.repeat(1, 3 - channel + 1, 1, 1)

        x = torch.nn.functional.pad(
            x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), value=0)

        # print('pad size: ', self.pad_size)
        # print('x shape: ', x.shape)

        base = torch.zeros(1, 3, self.base_size, self.base_size).to(x.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])
        x_prompted = x + self.p_eps * prompt

        return x_prompted


class Coordinator(nn.Module):
    def __init__(self, num_src):
        super(Coordinator, self).__init__()
        # self.args = args
        self.backbone = 'vit-mae-base'
        act = nn.GELU  # if args.TRAINER.ICLVP.ACT == 'gelu' else nn.ReLU

        src_dim = num_src

        z_dim = 768
        if self.backbone == 'vit-mae-base':  # ! SSL-MAE VIT-B (n param: 86M)
            self.enc_pt = ViTForImageClassification.from_pretrained("facebook/vit-mae-base")
        elif self.backbone == 'vit-base':  # ! SUP VIT-B
            self.enc_pt = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        elif self.backbone == 'dino-resnet-50':  # ! SSL-DINO RN50 (n param: 23M)
            self.enc_pt = ResNetModel.from_pretrained("Ramos-Ramos/dino-resnet-50")
            z_dim = 2048
        else:
            raise ValueError('not implemented')

        self.dec = DecoderManual(z_dim, src_dim, act=act, arch=self.backbone)

    def forward(self, x):
        # with torch.no_grad():
        if self.backbone == 'vit-mae-base':
            # ! (N, 197, 768) => pick [CLS] => (N, 768)
            out = self.enc_pt(x, output_hidden_states=True)
            z = out.hidden_states[-1][:, 0, :]
        elif self.backbone == 'vit-base':
            # ! (N, 197, 768) => pick [CLS] => (N, 768)
            out = self.enc_pt(x)
            z = out.last_hidden_state[:, 0, :]
            # z = out.last_hidden_state  # 使用所有 token 的特征
        elif self.backbone == 'dino-resnet-50':
            # ! (N, 2048, 7, 7) => pool => (N, 2048)
            out_temp = self.enc_pt(x)
            zdim_ = out_temp.last_hidden_state.shape[1]
            out = out_temp.pooler_output.reshape(-1, zdim_)
            z = out
        else:
            raise ValueError

        wrap = self.dec(z)
        return wrap, z


class DecoderManual(nn.Module):
    def __init__(self, i_dim, src_dim, act=nn.GELU, arch='vit-base'):
        super(DecoderManual, self).__init__()
        if i_dim:
            self.shared_feature = 1
        else:
            self.shared_feature = 0
        if self.shared_feature:
            # ! start from 7*7*16(784:16) or 7*7*32(1568:800) or 7*7*64(3,136:2368)
            if (src_dim % 49) != 0: raise ValueError('map dim must be devided with 7*7')
            self.p_trigger = torch.nn.Parameter(torch.randn(1, src_dim - i_dim))
            # torch.nn.init.uniform_(self.p_trigger, a=0.0, b=0.1)  # can be tuned
            src_c = src_dim // 49
        else:
            src_c = src_dim

        bias_flag = False
        body_seq = []

        if arch in ['vit-mae-base', 'vit-base']:
            if src_c >= 64:
                g_c = 64
            else:
                g_c = src_c
            body_seq += [nn.ConvTranspose2d(src_c, 64, 2, 2, 0, groups=g_c),
                         nn.ConvTranspose2d(64, 64, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(64), act()]
            body_seq += [nn.ConvTranspose2d(64, 64, 2, 2, 0, groups=64),
                         nn.ConvTranspose2d(64, 32, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(32), act()]
            body_seq += [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                         nn.ConvTranspose2d(32, 32, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(32), act()]
            body_seq += [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                         nn.ConvTranspose2d(32, 16, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(16), act()]
            body_seq += [nn.ConvTranspose2d(16, 3, 2, 2, 0, bias=bias_flag)]
        elif arch == 'dino-resnet-50':
            body_seq += [nn.ConvTranspose2d(src_c, 64, 2, 2, 0, groups=32),
                         nn.ConvTranspose2d(64, 64, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(64), act()]
            body_seq += [nn.ConvTranspose2d(64, 64, 2, 2, 0, groups=64),
                         nn.ConvTranspose2d(64, 32, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(32), act()]
            body_seq += [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                         nn.ConvTranspose2d(32, 32, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(32), act()]
            body_seq += [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                         nn.ConvTranspose2d(32, 16, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(16), act()]
            body_seq += [nn.ConvTranspose2d(16, 3, 2, 2, 0, bias=bias_flag)]
        else:
            raise ValueError('not implemented')
        self.body = nn.Sequential(*body_seq)

    def forward(self, z):
        if self.shared_feature:
            N = z.shape[0]
            D = self.p_trigger.shape[1]
            p_trigger = self.p_trigger.repeat(N, 1)
            z_cube = torch.cat((z, p_trigger), dim=1)
            z_cube = z_cube.reshape(N, -1, 7, 7)
        else:
            return self.body(z)
        return self.body(z_cube)


class EncoderManual(nn.Module):
    def __init__(self, out_dim, act=nn.GELU, gap=False):
        super(EncoderManual, self).__init__()
        bias_flag = False
        body_seq = []
        body_seq += [nn.Conv2d(3, 32, 3, 1, 1),
                     nn.Conv2d(32, 32, 2, 2, 0, bias=bias_flag)]
        body_seq += [nn.BatchNorm2d(32), act()]
        body_seq += [nn.Conv2d(32, 32, 3, 1, 1),
                     nn.Conv2d(32, 64, 2, 2, 0, bias=bias_flag)]
        body_seq += [nn.BatchNorm2d(64), act()]
        body_seq += [nn.Conv2d(64, 64, 3, 1, 1),
                     nn.Conv2d(64, 64, 2, 2, 0, bias=bias_flag)]
        body_seq += [nn.BatchNorm2d(64), act()]
        body_seq += [nn.Conv2d(64, 64, 3, 1, 1),
                     nn.Conv2d(64, 128, 2, 2, 0, bias=bias_flag)]
        body_seq += [nn.BatchNorm2d(128), act()]
        body_seq += [nn.Conv2d(128, 128, 3, 1, 1),
                     nn.Conv2d(128, out_dim, 2, 2, 0, bias=bias_flag)]
        body_seq += [nn.BatchNorm2d(out_dim), act()]
        if gap:     body_seq += [nn.AdaptiveAvgPool2d((1, 1))]
        self.body = nn.Sequential(*body_seq)

    def forward(self, x):
        return self.body(x)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attn = self.conv(x)
        attn = torch.sigmoid(attn)
        return x * attn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DecoderManualAttention(nn.Module):
    def __init__(self, i_dim, src_dim, act=nn.GELU, arch='vit-base'):
        super(DecoderManualAttention, self).__init__()
        if i_dim:
            self.shared_feature = 1
        else:
            self.shared_feature = 0
        if self.shared_feature:
            if (src_dim % 49) != 0: raise ValueError('map dim must be devided with 7*7')
            self.p_trigger = torch.nn.Parameter(torch.Tensor(1, src_dim - i_dim))
            torch.nn.init.uniform_(self.p_trigger, a=0.0, b=0.1)
            src_c = src_dim // 49
        else:
            src_c = src_dim

        bias_flag = False
        body_seq = []

        if arch in ['vit-base', 'vit-mae-base']:
            if src_c >= 64:
                g_c = 64
            else:
                g_c = src_c
            body_seq += [nn.ConvTranspose2d(src_c, 512, kernel_size=4, stride=2, padding=1, groups=g_c),
                         nn.ConvTranspose2d(512, 512, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(512), act()]
            body_seq += [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, groups=64),
                         nn.ConvTranspose2d(256, 256, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(256), act()]

            self.attn1 = SpatialAttention(256)
            self.attn2 = ChannelAttention(256)

            body_seq += [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, groups=32),
                         nn.ConvTranspose2d(128, 128, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(128), act()]

            self.attn3 = SpatialAttention(128)
            self.attn4 = ChannelAttention(128)

            body_seq += [nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, groups=16),
                         nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=bias_flag)]
        else:
            raise ValueError('not implemented')
        self.body = nn.Sequential(*body_seq)

    def forward(self, z):
        if self.shared_feature:
            N = z.shape[0]
            D = self.p_trigger.shape[1]
            p_trigger = self.p_trigger.repeat(N, 1)
            z_cube = torch.cat((z, p_trigger), dim=1)
            z_cube = z_cube.reshape(N, -1, 7, 7)
            z = self.body(z_cube)
        else:
            z = z.reshape(z.shape[0], -1, 7, 7)
            z = self.body[:4](z)
            z = self.attn1(z)
            z = self.attn2(z)
            z = self.body[4:8](z)
            z = self.attn3(z)
            z = self.attn4(z)
            z = self.body[8:](z)
        z = torch.sigmoid(z)
        return z


class DecoderManualPadding(nn.Module):
    def __init__(self, i_dim, src_dim, lam, act=nn.GELU, arch='vit-base'):
        super(DecoderManualPadding, self).__init__()
        if i_dim:
            self.shared_feature = 1
        else:
            self.shared_feature = 0
        if self.shared_feature:
            if (src_dim % 49) != 0: raise ValueError('map dim must be devided with 7*7')
            self.p_trigger = torch.nn.Parameter(torch.Tensor(1, src_dim - i_dim))
            torch.nn.init.uniform_(self.p_trigger, a=0.0, b=0.1)
            src_c = src_dim // 49
        else:
            src_c = src_dim

        bias_flag = False
        body_seq = []

        if arch in ['vit-mae-base', 'vit-base']:
            if src_c >= 64:
                g_c = 64
            else:
                g_c = src_c
            body_seq += [nn.ConvTranspose2d(src_c, 64, 2, 2, 0, groups=g_c),
                         nn.ConvTranspose2d(64, 64, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(64), act()]
            body_seq += [nn.ConvTranspose2d(64, 64, 2, 2, 0, groups=64),
                         nn.ConvTranspose2d(64, 32, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(32), act()]
            body_seq += [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                         nn.ConvTranspose2d(32, 32, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(32), act()]
            body_seq += [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                         nn.ConvTranspose2d(32, 16, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(16), act()]
            body_seq += [nn.ConvTranspose2d(16, 3, 2, 2, 0, bias=bias_flag)]
        elif arch == 'dino-resnet-50':
            body_seq += [nn.ConvTranspose2d(src_c, 64, 2, 2, 0, groups=32),
                         nn.ConvTranspose2d(64, 64, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(64), act()]
            body_seq += [nn.ConvTranspose2d(64, 64, 2, 2, 0, groups=64),
                         nn.ConvTranspose2d(64, 32, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(32), act()]
            body_seq += [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                         nn.ConvTranspose2d(32, 32, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(32), act()]
            body_seq += [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                         nn.ConvTranspose2d(32, 16, kernel_size=1, bias=bias_flag)]
            body_seq += [nn.BatchNorm2d(16), act()]
            body_seq += [nn.ConvTranspose2d(16, 3, 2, 2, 0, bias=bias_flag)]
        else:
            raise ValueError('not implemented')
        self.body = nn.Sequential(*body_seq)

        # 添加随机初始化的padding噪声
        image_size = 224
        self.pad_size = 30
        self.base_size = image_size - self.pad_size * 2
        self.pad_up = nn.Parameter(torch.randn([1, 3, self.pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, self.pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - self.pad_size * 2, self.pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - self.pad_size * 2, self.pad_size]))

        if lam == 'random':
            self.lam = nn.Parameter(torch.tensor(0.5))
        else:
            self.lam = float(lam)

    def forward(self, z):
        if self.shared_feature:
            N = z.shape[0]
            D = self.p_trigger.shape[1]
            p_trigger = self.p_trigger.repeat(N, 1)
            z_cube = torch.cat((z, p_trigger), dim=1)
            z_cube = z_cube.reshape(N, -1, 7, 7)
            z = self.body(z_cube)
        else:
            z = self.body(z)
        # print('z shape: ', z.shape)  # torch.Size([32, 3, 224, 224])

        # 添加padding噪声
        base = torch.zeros(1, 3, self.base_size, self.base_size).to(z.device)
        pad = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        pad = torch.cat([self.pad_up, pad, self.pad_down], dim=2)
        pad = torch.cat(z.size(0) * [pad])

        z = self.lam * z + (1 - self.lam) * pad
        # z = torch.sigmoid(z)

        return z


class DecoderOnlyPadding(nn.Module):
    def __init__(self, i_dim, src_dim, lam, act=nn.GELU, arch='vit-base'):
        super(DecoderOnlyPadding, self).__init__()

        # 添加随机初始化的padding噪声
        image_size = 224
        self.pad_size = 30
        self.base_size = image_size - self.pad_size * 2
        self.pad_up = nn.Parameter(torch.randn([1, 3, self.pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, self.pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - self.pad_size * 2, self.pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - self.pad_size * 2, self.pad_size]))

        if lam == 'random':
            self.lam = nn.Parameter(torch.tensor(0.5))
        else:
            self.lam = float(lam)

        # 添加线性层,将768维特征转换为3 * base_size * base_size维
        self.linear = nn.Linear(i_dim, 3 * self.base_size * self.base_size)

    def forward(self, z):
        # print('z shape: ', z.shape)  # torch.Size([32, 768])

        # 使用线性层将768维特征转换为3 * base_size * base_size维
        z = self.linear(z)
        z = z.view(-1, 3, self.base_size, self.base_size)

        # 调整self.pad_left、self.pad_right、self.pad_up和self.pad_down的尺寸,使其在维度0上与z的batch size相匹配
        pad_left = self.pad_left.repeat(z.size(0), 1, 1, 1)
        pad_right = self.pad_right.repeat(z.size(0), 1, 1, 1)
        pad_up = self.pad_up.repeat(z.size(0), 1, 1, 1)
        pad_down = self.pad_down.repeat(z.size(0), 1, 1, 1)

        # 添加padding噪声
        pad = torch.cat([pad_left, z, pad_right], dim=3)
        pad = torch.cat([pad_up, pad, pad_down], dim=2)
        # z = self.lam * z + (1 - self.lam) * pad

        return pad


class CoordinatorINIT(nn.Module):
    def __init__(self):
        super(CoordinatorINIT, self).__init__()

        act = nn.GELU  # if args.TRAINER.BLACKVIP.ACT == 'gelu' else nn.ReLU
        e_out_dim = 768
        src_dim = 768

        self.enc = EncoderManual(e_out_dim, act=act, gap=False)
        self.dec = DecoderManual(0, src_dim=e_out_dim, act=act, arch='vit-base')

    def forward(self, x):
        z = self.enc(x)
        wrap = self.dec(z)
        return wrap, z


def coordinator(num_src):
    return Coordinator(num_src)


def coordinator_init():
    return CoordinatorINIT()
