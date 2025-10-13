import math

import torch
import torch.nn as nn
from vector_quantize_pytorch import FSQ

from .utils import weight_init


class MLPBlock(nn.Module):
    def __init__(self, dim, expand=4, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, expand * dim),
            nn.ReLU6(),
            nn.Linear(expand * dim, dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.mlp(x))


class LatentActHead(nn.Module):
    def __init__(self, act_dim, emb_dim, hidden_dim, expand=4, dropout=0.0):
        super().__init__()
        self.proj0 = nn.Linear(2 * emb_dim, hidden_dim)
        self.proj1 = nn.Linear(hidden_dim + 2 * emb_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim + 2 * emb_dim, hidden_dim)
        self.proj_end = nn.Linear(hidden_dim, act_dim)

        self.block0 = MLPBlock(hidden_dim, expand, dropout)
        self.block1 = MLPBlock(hidden_dim, expand, dropout)
        self.block2 = MLPBlock(hidden_dim, expand, dropout)

    def forward(self, obs_emb, next_obs_emb):
        x = self.block0(self.proj0(torch.concat([obs_emb, next_obs_emb], dim=-1)))
        x = self.block1(self.proj1(torch.concat([x, obs_emb, next_obs_emb], dim=-1)))
        x = self.block2(self.proj2(torch.concat([x, obs_emb, next_obs_emb], dim=-1)))
        x = self.proj_end(x)
        return x


class LatentObsHead(nn.Module):
    def __init__(self, act_dim, proj_dim, hidden_dim, expand=4, dropout=0.0):
        super().__init__()
        self.proj0 = nn.Linear(act_dim + proj_dim, hidden_dim)
        self.proj1 = nn.Linear(act_dim + hidden_dim, hidden_dim)
        self.proj2 = nn.Linear(act_dim + hidden_dim, hidden_dim)
        self.proj_end = nn.Linear(hidden_dim, proj_dim)

        self.block0 = MLPBlock(hidden_dim, expand, dropout)
        self.block1 = MLPBlock(hidden_dim, expand, dropout)
        self.block2 = MLPBlock(hidden_dim, expand, dropout)

    def forward(self, x, action):
        x = self.block0(self.proj0(torch.concat([x, action], dim=-1)))
        x = self.block1(self.proj1(torch.concat([x, action], dim=-1)))
        x = self.block2(self.proj2(torch.concat([x, action], dim=-1)))
        x = self.proj_end(x)
        return x


# inspired by:
# 1. https://github.com/schmidtdominik/LAPO/blob/main/lapo/models.py
# 2. https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU6(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU6(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, input_shape, out_channels, num_res_blocks=2, dropout=0.0, downscale=True):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self._downscale = downscale
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
            stride=2 if self._downscale else 1,
        )
        # conv downsampling is faster that maxpool, with same perf
        # self.conv = nn.Conv2d(
        #     in_channels=self._input_shape[0],
        #     out_channels=self._out_channels,
        #     kernel_size=3,
        #     padding=1,
        # )
        self.blocks = nn.Sequential(*[ResidualBlock(self._out_channels, dropout) for _ in range(num_res_blocks)])

    def forward(self, x):
        x = self.conv(x)
        # x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.blocks(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        if self._downscale:
            return (self._out_channels, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self._out_channels, h, w)


class DecoderBlock(nn.Module):
    def __init__(self, input_shape, out_channels, num_res_blocks=2):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels

        # upsample + conv works fine, just slower than conv-transpose
        # also: upsample does not work well with orthogonal init (why?)!
        # self.conv = nn.Conv2d(
        #     in_channels=self._input_shape[0],
        #     out_channels=self._out_channels,
        #     kernel_size=3,
        #     padding=1,
        # )
        self.conv = nn.ConvTranspose2d(
            in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=2, stride=2
        )
        self.blocks = nn.Sequential(*[ResidualBlock(self._out_channels) for _ in range(num_res_blocks)])

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=2)
        x = self.conv(x)
        x = self.blocks(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, h * 2, w * 2)


class Actor(nn.Module):
    def __init__(
        self,
        shape,
        num_actions,
        encoder_scale=1,
        encoder_channels=(16, 32, 32),
        encoder_num_res_blocks=1,
        dropout=0.0,
    ):
        super().__init__()
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.final_encoder_shape = shape
        self.encoder = nn.Sequential(
            *conv_stack,
            # nn.Flatten(),
        )
        self.actor_mean = nn.Sequential(
            nn.ReLU6(),
            # works either way...
            # nn.Linear(math.prod(shape), num_actions),
            nn.Linear(shape[0], num_actions),
        )
        self.num_actions = num_actions
        self.apply(weight_init)

    def forward(self, obs):
        out = self.encoder(obs)
        out = out.flatten(2).mean(-1)
        act = self.actor_mean(out)
        return act, out


class ActionDecoder(nn.Module):
    def __init__(self, obs_emb_dim, latent_act_dim, true_act_dim, hidden_dim=128):
        super().__init__()
        self.obs_emb_dim = obs_emb_dim
        self.latent_act_dim = latent_act_dim
        self.true_act_dim = true_act_dim

        self.model = nn.Sequential(
            # nn.Linear(latent_act_dim + obs_emb_dim, hidden_dim)
            nn.Linear(latent_act_dim, hidden_dim),
            nn.ReLU6(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU6(),
            nn.Linear(hidden_dim, true_act_dim),
        )

    def forward(self, obs_emb, latent_act):
        # hidden = torch.concat([obs_emb, latent_act], dim=-1)
        # true_act_pred = self.model(hidden)
        true_act_pred = self.model(latent_act)
        return true_act_pred


# IDM: (s_t, s_t+1) -> a_t
class IDM(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
    ):
        super().__init__()
        shape = (shape[0] * 2, *shape[1:])
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(in_features=math.prod(shape), out_features=latent_act_dim),
            # nn.LayerNorm(latent_act_dim),
        )

    def forward(self, obs, next_obs):
        # [B, C, H, W] -> [B, 2 * C, H, W]
        concat_obs = torch.concat([obs, next_obs], axis=1)
        latent_action = self.encoder(concat_obs)
        return latent_action


# FDM: (s_t, a_t) -> s_t+1
class FDM(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
    ):
        super().__init__()
        self.inital_shape = shape

        # encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(*conv_stack)
        self.final_encoder_shape = shape

        # decoder
        shape = (shape[0] * 2, *shape[1:])
        conv_stack = []
        for out_ch in encoder_channels[::-1]:
            conv_seq = DecoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.decoder = nn.Sequential(
            *conv_stack,
            nn.GELU(),
            nn.Conv2d(encoder_channels[0] * encoder_scale, self.inital_shape[0], kernel_size=1),
            nn.Tanh(),
        )
        self.act_proj = nn.Linear(latent_act_dim, math.prod(self.final_encoder_shape))

    def forward(self, obs, latent_action):
        assert obs.ndim == 4, "expect shape [B, C, H, W]"
        obs_emb = self.encoder(obs)
        act_emb = self.act_proj(latent_action).reshape(-1, *self.final_encoder_shape)
        # concat across channels, [B, C * 2, 1, 1]
        emb = torch.concat([obs_emb, act_emb], dim=1)
        next_obs = self.decoder(emb)
        return next_obs


class LAPO(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
    ):
        super().__init__()
        self.idm = IDM(
            shape=shape,
            latent_act_dim=latent_act_dim,
            encoder_scale=encoder_scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks,
        )
        self.fdm = FDM(
            shape=shape,
            latent_act_dim=latent_act_dim,
            encoder_scale=encoder_scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks,
        )
        self.latent_act_dim = latent_act_dim
        self.apply(weight_init)

    def forward(self, obs, next_obs):
        latent_action = self.idm(obs, next_obs)
        next_obs_pred = self.fdm(obs, latent_action)
        return next_obs_pred, latent_action

    @torch.no_grad()
    def label(self, obs, next_obs):
        latent_action = self.idm(obs, next_obs)
        return latent_action


# Not used in final experiments, here just for reference.
class IDMFSQ(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim=128,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        fsq_levels=(2, 2),
    ):
        super().__init__()
        assert latent_act_dim % len(fsq_levels) == 0
        self.latent_act_dim = latent_act_dim
        self.fsq_levels = fsq_levels

        shape = (shape[0] * 2, *shape[1:])
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(in_features=math.prod(shape), out_features=latent_act_dim),
            # nn.LayerNorm(latent_act_dim),
        )
        self.quantizer = FSQ(levels=list(fsq_levels))

    def forward(self, obs, next_obs):
        # [B, C, H, W] -> [B, 2 * C, H, W]
        concat_obs = torch.concat([obs, next_obs], axis=1)
        # [B, la_dim]
        latent_action = self.encoder(concat_obs)
        # [B, la_split, la_dim // la_split]
        latent_action = latent_action.reshape(latent_action.shape[0], self.latent_act_dim // len(self.fsq_levels), -1)
        quantized_latent_action, indices = self.quantizer(latent_action)
        quantized_latent_action = quantized_latent_action.reshape(concat_obs.shape[0], -1)
        assert quantized_latent_action.shape[-1] == self.latent_act_dim

        return quantized_latent_action


class LAPOFSQ(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim=128,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        fsq_levels=(2, 2, 4),
    ):
        super().__init__()
        self.idm = IDMFSQ(
            shape=shape,
            latent_act_dim=latent_act_dim,
            encoder_scale=encoder_scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks,
            fsq_levels=fsq_levels,
        )
        self.fdm = FDM(
            shape=shape,
            latent_act_dim=latent_act_dim,
            encoder_scale=encoder_scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks,
        )
        self.latent_act_dim = latent_act_dim
        self.apply(weight_init)

    def forward(self, obs, next_obs):
        latent_action = self.idm(obs, next_obs)
        next_obs_pred = self.fdm(obs, latent_action)
        return next_obs_pred, latent_action

    @torch.no_grad()
    def label(self, obs, next_obs):
        latent_action = self.idm(obs, next_obs)
        return latent_action


class LAOM(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        encoder_dropout=0.0,
        encoder_norm_out=True,
        act_head_dim=512,
        act_head_dropout=0.0,
        obs_head_dim=512,
        obs_head_dropout=0.0,
    ):
        super().__init__()
        self.inital_shape = shape

        # encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, encoder_dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.LayerNorm(math.prod(shape), elementwise_affine=False) if encoder_norm_out else nn.Identity(),
        )
        self.act_head = LatentActHead(latent_act_dim, math.prod(shape), act_head_dim, dropout=act_head_dropout)
        self.obs_head = LatentObsHead(latent_act_dim, math.prod(shape), obs_head_dim, dropout=obs_head_dropout)
        self.final_encoder_shape = shape
        self.latent_act_dim = latent_act_dim
        self.apply(weight_init)

    def forward(self, obs, next_obs):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])

        latent_action = self.act_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        latent_next_obs = self.obs_head(obs_emb.flatten(1).detach(), latent_action)

        return latent_next_obs, latent_action, obs_emb.detach()

    @torch.no_grad()
    def label(self, obs, next_obs):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        latent_action = self.act_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        return latent_action


class LAOMWithLabels(nn.Module):
    def __init__(
        self,
        shape,
        true_act_dim,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        encoder_dropout=0.0,
        encoder_norm_out=True,
        act_head_dim=512,
        act_head_dropout=0.0,
        obs_head_dim=512,
        obs_head_dropout=0.0,
    ):
        super().__init__()
        self.inital_shape = shape

        # encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, encoder_dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.LayerNorm(math.prod(shape), elementwise_affine=False) if encoder_norm_out else nn.Identity(),
        )
        self.idm_head = LatentActHead(latent_act_dim, math.prod(shape), act_head_dim, dropout=act_head_dropout)
        self.true_actions_head = nn.Linear(latent_act_dim, true_act_dim)

        self.fdm_head = LatentObsHead(latent_act_dim, math.prod(shape), obs_head_dim, dropout=obs_head_dropout)
        self.final_encoder_shape = shape
        self.latent_act_dim = latent_act_dim
        self.apply(weight_init)

    def forward(self, obs, next_obs, predict_true_act=False):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])

        latent_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        latent_next_obs = self.fdm_head(obs_emb.flatten(1).detach(), latent_action)
        # TODO: use norm from encoder here too!

        if predict_true_act:
            true_action = self.true_actions_head(latent_action)
            return latent_next_obs, latent_action, true_action, obs_emb.flatten(1).detach()

        return latent_next_obs, latent_action, obs_emb.flatten(1).detach()

    @torch.no_grad()
    def label(self, obs, next_obs):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        latent_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        return latent_action


class IDMLabels(nn.Module):
    def __init__(
        self,
        shape,
        act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        encoder_dropout=0.0,
        act_head_dim=512,
        act_head_dropout=0.0,
    ):
        super().__init__()
        self.inital_shape = shape

        # encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, encoder_dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            # nn.LayerNorm(math.prod(shape))
        )
        self.idm_head = LatentActHead(act_dim, math.prod(shape), act_head_dim, dropout=act_head_dropout)

        self.act_dim = act_dim
        self.final_encoder_shape = shape
        self.apply(weight_init)

    def forward(self, obs, next_obs):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        pred_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))

        return pred_action, obs_emb.flatten(1).detach()

    @torch.no_grad()
    def label(self, obs, next_obs):
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        pred_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        return pred_action
