from models.mossformer_gan.fsmn import UniDeepFsmn
from models.mossformer_gan.conv_module import ConvModule
from models.mossformer_gan.mossformer import MossFormer
from models.mossformer_gan.se_layer import SELayer
from models.mossformer_gan.get_layer_from_string import get_layer
from models.mossformer_gan.discriminator import Discriminator
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import parse as V
from torch.nn import init
from torch.nn.parameter import Parameter

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")

class MossFormerGAN_SE_16K(nn.Module):
    def __init__(self, args):
        super(MossFormerGAN_SE_16K, self).__init__()
        self.model = SyncANet(num_channel=64, num_features=args.fft_len//2+1)
        if args.mode == 'train':
            self.discriminator = Discriminator(ndf=16)
        else:
            self.discriminator = None

    def forward(self, x):
        output_real, output_imag = model(x)
        return output_real, output_imag

class FSMN_Wrap(nn.Module):

    def __init__(self, nIn, nHidden=128, lorder=20, nOut=128):
        super(FSMN_Wrap, self).__init__()

        self.fsmn = UniDeepFsmn(nIn, nHidden, lorder, nHidden)

    def forward(self, x):
        # # shpae of input x : [b,c,h,T,2], [6, 256, 1, 106]
        b,c,T,h = x.size()
        #x : [b,T,h,c]
        x = x.permute(0, 2, 3, 1)
        x = torch.reshape(x, (b*T, h, c))

        output = self.fsmn(x)
        # output: [b*T,h,c], [6*106, h, 256]

        output = torch.reshape(output, (b, T, h, c))
        # output: [b,c,h,T], [6, 99, 1024]

        return output.permute(0, 3, 1, 2)

class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))
            setattr(self, 'fsmn{}'.format(i + 1), FSMN_Wrap(nIn=self.in_channels, nHidden=self.in_channels, lorder=5, nOut=self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            out = getattr(self, 'fsmn{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x

class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1)


class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        return x


class SyncANet(nn.Module):
    def __init__(self, num_channel=64, num_features=201):
        super(SyncANet, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)
        self.n_layers = 6
        self.blocks = nn.ModuleList([])
        for _ in range(self.n_layers):
            self.blocks.append(
                SyncANetBlock(
                    emb_dim=num_channel,
                    emb_ks=2,
                    emb_hs=1,
                    n_freqs=int(num_features//2)+1,
                    hidden_channels=num_channel*2,
                    n_head=4,
                    approx_qk_dim=512,
                    activation='prelu',
                    eps=1.0e-5,
                )
            )

        self.mask_decoder = MaskDecoder(num_features, num_channel=num_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)

    def forward(self, x):
        out_list = []
        mag = torch.sqrt(x[:, 0, :, :]**2 + x[:, 1, :, :]**2).unsqueeze(1)
        noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1)

        x = self.dense_encoder(x_in)
        for ii in range(self.n_layers):
            x = self.blocks[ii](x)

        mask = self.mask_decoder(x)
        out_mag = mask * mag

        complex_out = self.complex_decoder(x)
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)
        out_list.append(final_real)
        out_list.append(final_imag)

        return out_list

class FFConvM(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass = nn.LayerNorm,
        dropout = 0.1
    ):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out),
            nn.Dropout(dropout)
        )
    def forward(
        self,
        x,
    ):
        output = self.mdl(x)
        return output

class SyncANetBlock(nn.Module):
    """
    The design of this block is partially motivated by TF-GridNet (https://arxiv.org/abs/2211.12433)
    We apply a modided MossFormer module (GatedFormer module) and FSMN moduels instead of RNN modules for computational efficiency and better performance
    Each GatedFormer module applies a triple-attention scheme
    """
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()

        in_channels = emb_dim * emb_ks

        ##intra modules
        self.Fconv = nn.Conv2d(emb_dim, in_channels, kernel_size=(1, emb_ks), stride=(1,1), groups=emb_dim)
        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_to_u = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_channels,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        self.intra_to_v = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_channels,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        self.intra_rnn = self._build_repeats(in_channels, hidden_channels, 20, hidden_channels, repeats=1)
        self.intra_mossformer = MossFormer(dim=emb_dim, group_size=n_freqs)

        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels, emb_dim, emb_ks, stride=emb_hs
        )
        self.intra_se = SELayer(channel=emb_dim, reduction=1)

        ##inter modules
        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_to_u = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_channels,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        self.inter_to_v = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_channels,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        self.inter_rnn = self._build_repeats(in_channels, hidden_channels, 20, hidden_channels, repeats=1)
        self.inter_mossformer = MossFormer(dim=emb_dim, group_size=256)

        self.inter_linear = nn.ConvTranspose1d(
            hidden_channels, emb_dim, emb_ks, stride=emb_hs
        )
        self.inter_se = SELayer(channel=emb_dim, reduction=1)

        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate
        assert emb_dim % n_head == 0
        for ii in range(n_head):
            self.add_module(
                "attn_conv_Q_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_K_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_V_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                ),
            )
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for i in range(repeats)
        ]
        return nn.Sequential(*repeats)
    
    def forward(self, x):
        """SyncANetBlock Forward.
        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # intra process
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = self.Fconv(intra_rnn) #[B, C*emb_ks, T, Q-emb_ks+1]
        intra_rnn = (
            intra_rnn.transpose(1, 2).contiguous().view(B * T, C*self.emb_ks, -1)
        )  # [BT, C, Q]

        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn_u = self.intra_to_u(intra_rnn)
        intra_rnn_v = self.intra_to_v(intra_rnn)
        intra_rnn_u = self.intra_rnn(intra_rnn_u)  # [BT, -1, H]        
        intra_rnn = intra_rnn_v * intra_rnn_u
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.transpose(1, 2)
        ##[BT, Q, C]        
        intra_rnn = intra_rnn.view([B, T, Q, C])
        intra_rnn = self.intra_mossformer(intra_rnn)
        intra_rnn = intra_rnn.transpose(1, 2)
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = self.intra_se(intra_rnn)
        intra_rnn = intra_rnn + input_  # [B, C, T, Q]

        # inter process
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, C, T, F]
        inter_rnn = (
            inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        )  # [BF, C, T]
        inter_rnn = F.unfold(
            inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BF, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        inter_rnn_u = self.inter_to_u(inter_rnn)
        inter_rnn_v = self.inter_to_v(inter_rnn)
        inter_rnn_u = self.inter_rnn(inter_rnn_u)  # [BF, -1, H]
        inter_rnn = inter_rnn_v * inter_rnn_u
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        inter_rnn = inter_rnn.transpose(1, 2)
        ## [BF, T, C]
        inter_rnn = inter_rnn.view([B, Q, T, C])
        inter_rnn = self.inter_mossformer(inter_rnn)
        inter_rnn = inter_rnn.transpose(1, 2)
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = self.inter_se(inter_rnn)
        inter_rnn = inter_rnn + input_  # [B, C, T, Q]

        # attention
        inter_rnn = inter_rnn[..., :old_T, :old_Q]

        batch = inter_rnn

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](batch))  # [B, C, T, Q]
            all_K.append(self["attn_conv_K_%d" % ii](batch))  # [B, C, T, Q]
            all_V.append(self["attn_conv_V_%d" % ii](batch))  # [B, C, T, Q]

        Q = torch.cat(all_Q, dim=0)  # [B', C, T, Q]
        K = torch.cat(all_K, dim=0)  # [B', C, T, Q]
        V = torch.cat(all_V, dim=0)  # [B', C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, C*Q]
        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])  # [n_head, B, C, T, Q])
        batch = batch.transpose(0, 1)  # [B, n_head, C, T, Q])
        batch = batch.contiguous().view(
            [B, self.n_head * emb_dim, old_T, -1]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        out = batch + inter_rnn
        return out


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat
