import functools
import logging
import os

# The above cannot be `from sade.models` as that does not populate the global _MODELS dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.segresnet_block import get_upsample_layer

from . import ema, layers, layerspp, registry

default_initializer = layers.default_init
MultiSequential = layers.MultiSequential
AttentionBlock = layerspp.ChannelAttentionBlock3d
get_upsample_layer_pp = functools.partial(
    get_upsample_layer, spatial_dims=3, upsample_mode="nontrainable"
)
get_conv_layer_pp = functools.partial(
    layerspp.get_conv_layer, spatial_dims=3, init_scale=0.0
)
ResBlockpp = functools.partial(layerspp.ResnetBlockBigGANpp, act="memswish", init_scale=0.0)


def load_model_from_config(config):
    model_name = config.model.name
    inner_model = registry.get_model(model_name)(config)
    checkpoint_path = config.training.pretrained_checkpoint
    state = dict(model=inner_model, step=0)
    state = restore_pretrained_weights(checkpoint_path, state, config.device)
    return state["model"]


def restore_pretrained_weights(ckpt_dir, state, device):
    assert state["step"] == 0, "Can only load pretrained weights when starting a new run"
    assert os.path.exists(ckpt_dir), f"Pretrain weights directory {ckpt_dir} does not exist"
    assert state[
        "model"
    ].training, "Model must be in training mode to appropriately load pretrained weights"

    loaded_state = torch.load(ckpt_dir, map_location=device)
    state["model_checkpoint_step"] = loaded_state["step"]
    dummy_ema = ema.ExponentialMovingAverage(state["model"].parameters(), decay=0.999)
    dummy_ema.load_state_dict(loaded_state["ema"])
    dummy_ema.lazy_copy_to(state["model"].parameters())
    logging.info(f"Loaded pretrained EMA weights from {ckpt_dir} at {loaded_state['step']}")

    return state


# Magnitude-preserving concatenation (Equation 103) from EDM2 paper.
def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t**2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a, wb * b], dim=dim)


@registry.register_model(name="single-matryoshka")
class SingleMatryoshka(registry.BaseScoreModel):
    """
    Single-cascade Matryoshka style model that uses a pre-trained lower resolution model as the inner unet
    Outer model is two ResNet blocks
    """

    def __init__(self, config):
        super().__init__(config)

        # standalone config of pre-trained inner unet
        self.inner_model_config = config.inner_model
        self.data_channels = config.data.num_channels
        self.inner_unet = load_model_from_config(self.inner_model_config)

        if not config.model.trainable_inner_model:
            self.inner_unet.requires_grad_(False)
            # Rescaling will be done by outer model
            self.inner_unet.scale_by_sigma = False

        self.init_filters = config.model.nf
        self.pool = torch.nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
        self.pad = functools.partial(torch.nn.functional.pad, pad=(0, 0, 4, 4, 4, 4))
        self.inner_size = self.inner_model_config.data.image_size
        # self.pool = functools.partial(torch.nn.functional.interpolate,
        #     size=self.inner_model_config.data.image_size
        # )
        self.blocks_down = 1  # config.model.blocks_down[0]
        self.blocks_up = 1  # config.model.blocks_up[0]

        # (2 *sin+cos)
        self.time_embedding_features = config.model.time_embedding_sz * 4
        self.learnable_embedding = config.model.learnable_embedding
        self.time_embedding_sz = config.model.time_embedding_sz
        self.fourier_scale = config.model.fourier_scale
        self.time_embed_layer = self._make_time_cond_layers("fourier")

        ###### Downsampling / Encoders #######
        self.enc = nn.ModuleDict()
        self.channel_expand = get_conv_layer_pp(
            in_channels=self.data_channels,
            out_channels=self.init_filters,
        )

        cin = self.init_filters
        cout = cin * 2

        self.enc["outer_dblock_preconv"] = get_conv_layer_pp(
            in_channels=cin, out_channels=cout
        )
        self.enc["outer_dblock_res-skip"] = ResBlockpp(
            cout, downsample=False, temb_dim=self.time_embedding_features
        )
        self.enc["outer_dblock_res-down"] = ResBlockpp(
            cout, downsample=True, temb_dim=self.time_embedding_features
        )

        self.channel_squeeze = get_conv_layer_pp(
            in_channels=cout, out_channels=self.data_channels
        )

        ###### Upsampling / Decoders #######
        cin = self.data_channels
        # cout is the final number of channels from encoder

        self.dec = nn.ModuleDict()
        self.dec["up_channel_expand"] = get_conv_layer_pp(
            in_channels=self.data_channels,
            out_channels=cout,
        )
        self.dec["outer_upblock_upsample"] = get_upsample_layer_pp(in_channels=cin)
        # There will be skip connection with encoder via concatenation
        cout *= 2
        self.dec["outer_upblock_res-skip"] = ResBlockpp(
            cout, downsample=False, temb_dim=self.time_embedding_features
        )
        self.dec["channel_squeeze"] = get_conv_layer_pp(
            in_channels=cout,
            out_channels=self.data_channels,
        )

    def _make_time_cond_layers(self, embedding_type):
        layer_list = []

        if embedding_type == "fourier":
            # Projection layer doubles the input_sz
            # Since it concats sin and cos projections
            projection = layerspp.GaussianFourierProjection(
                embedding_size=self.time_embedding_sz,
                scale=self.fourier_scale,
                learnable=self.learnable_embedding,
            )
            layer_list.append(projection)

        sz = self.time_embedding_sz * 2
        dense_0 = layerspp.make_dense_layer(sz, sz * 2)
        dense_1 = layerspp.make_dense_layer(sz * 2, sz * 2)

        layer_list.append(dense_0)
        layer_list.append(nn.SiLU())
        layer_list.append(dense_1)

        return nn.Sequential(*layer_list)

    def forward(self, xin, sigma, out=None):
        # x = fup ( finner ( fdown(x) ) )
        temb = self.time_embed_layer(torch.log(sigma))

        # starting outer unet coomputation
        x = self.channel_expand(xin)
        res_skips = []
        x = self.enc["outer_dblock_preconv"](x)

        x = self.enc["outer_dblock_res-skip"](x, temb)
        res_skips.append(x)
        x = self.enc["outer_dblock_res-down"](x, temb)
        # x = torch.utils.checkpoint.checkpoint(
        #     self.enc["outer_dblock_res-down"], *(x, temb), use_reentrant=True
        #     )

        # preparing to ingest input to inner unet
        x = self.channel_squeeze(x)
        sz = x.shape[2:]
        xin_inner = self.pool(xin)
        x += xin_inner
        x = F.interpolate(x, size=self.inner_size)

        # a full-fledged unet
        sigma_rescale_factor = np.sqrt(np.prod(x.shape[2:]) / np.prod(sz))
        x = self.inner_unet(x, sigma * sigma_rescale_factor)
        x = F.interpolate(x, size=sz)

        if out is not None:
            out.append((xin_inner, x))

        x = self.dec["up_channel_expand"](x)
        x = self.dec["outer_upblock_upsample"](x)
        x = mp_cat(x, res_skips.pop())
        x = self.dec["outer_upblock_res-skip"](x, temb)
        x = self.dec["channel_squeeze"](x)

        if out is not None:
            out.append((xin, x))

        # scale_by_sigma
        sigma = sigma.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
        x = x / sigma

        return x
