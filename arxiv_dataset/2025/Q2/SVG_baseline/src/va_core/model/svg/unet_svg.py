from functools import partial
import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F

from diffusers.models.modeling_utils import ModelMixin
from diffusers.models import UNet3DConditionModel
from diffusers.configuration_utils import ConfigMixin, register_to_config, FrozenDict
from diffusers.loaders import UNet2DConditionLoadersMixin


from .modules_video import SpatioTemporalAttention
from .modules_attention import SpatialTransformer
from .modules_others import TimestepEmbedSequential, ResBlock, Downsample, normalization
from .modules_original import AdditiveConditionTransformer


class UNetModelSVG(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 unet_audio_config={},
                 unet_video_config={},  
                 connector_audio_config={},
                 connector_video_config={}):
        super().__init__()

        # Initialize connectors
        # connector_config should include the following values.
        # For in:
        #   with_connector: [bool] * length(unet.down_blocks)
        #   connector_channel: int
        #   connector_in_type: str ("attn" or "additive")
        # For out:
        #   channel_mult_connector: [int] * n
        #   num_blocks_connector: [int] * n
        #   connector_channel: int (same with that used for in)

        # default parameters
        tmp = [False, False, True, True]
        if "with_connector" not in connector_audio_config:
            connector_audio_config["with_connector"] = tmp
        if "with_connector" not in connector_video_config:
            connector_video_config["with_connector"] = tmp

        if "connector_channel" not in connector_audio_config:
            connector_audio_config["connector_channel"] = 1280
        if "connector_channel" not in connector_video_config:
            connector_video_config["connector_channel"] = 1280

        if "channel_mult_connector" not in connector_audio_config:
            connector_audio_config["channel_mult_connector"] = [2 ** (n+1) for n in range(sum(connector_audio_config["with_connector"]))]
        if "channel_mult_connector" not in connector_video_config:
            connector_video_config["channel_mult_connector"] = [2 ** (n+1) for n in range(sum(connector_video_config["with_connector"]))]

        if "num_blocks_connector" not in connector_audio_config:
            connector_audio_config["num_blocks_connector"] = [1,] * sum(connector_audio_config["with_connector"])
        if "num_blocks_connector" not in connector_video_config:
            connector_video_config["num_blocks_connector"] = [1,] * sum(connector_video_config["with_connector"])

        if "connector_in_type" not in connector_audio_config:
            connector_audio_config["connector_in_type"] = "attn"
        if "connector_in_type" not in connector_video_config:
            connector_video_config["connector_in_type"] = "attn"

        if "connector_out_input" not in connector_audio_config:
            connector_audio_config["connector_out_input"] = "x_pred"
        if "connector_out_input" not in connector_video_config:
            connector_video_config["connector_out_input"] = "x_pred"        

        if isinstance(unet_video_config, dict):
            unet_video_config = FrozenDict(unet_video_config)
        if isinstance(unet_audio_config, dict):
            unet_audio_config = FrozenDict(unet_audio_config)

        # connector_in
        self.connector_in_video = self._setup_connector_in(unet_video_config, connector_video_config)
        self.connector_in_audio = self._setup_connector_in(unet_audio_config, connector_audio_config)

        # connector_out
        self.connector_out_video = self._setup_connector_out(unet_video_config, connector_video_config, is_video=True)
        self.connector_out_audio = self._setup_connector_out(unet_audio_config, connector_audio_config)

        self.connector_audio_config = connector_audio_config
        self.connector_video_config = connector_video_config


    def _setup_connector_in(self, unet_config, connector_config):
        connector_in_down = []
        connector_in_up = []
        out_ch = connector_config["connector_channel"]
        for i in range(len(connector_config["with_connector"])):
            # for down_blocks
            cur_ch = unet_config.block_out_channels[i]
            dim_head = unet_config.attention_head_dim
            num_heads = cur_ch // dim_head
            if connector_config["with_connector"][i]:
                if connector_config["connector_in_type"] == "attn":
                    layer = TimestepEmbedSequential(*[SpatialTransformer(cur_ch, num_heads, dim_head, depth=1, context_dim=out_ch)])
                elif connector_config["connector_in_type"] == "additive":
                    axis_to_align_x = 2
                    axis_to_align_c = 2
                    layer = TimestepEmbedSequential(*[AdditiveConditionTransformer(cur_ch, num_heads, dim_head, axis_to_align_x, axis_to_align_c, depth=1, context_dim=out_ch)])
            else:
                layer = None
            connector_in_down.append(layer)

        for i in range(len(connector_config["with_connector"])):
            # for up_blocks
            cur_ch = unet_config.block_out_channels[-(i+1)]
            dim_head = unet_config.attention_head_dim
            num_heads = cur_ch // dim_head
            if connector_config["with_connector"][-(i+1)]:
                if connector_config["connector_in_type"] == "attn":
                    layer = TimestepEmbedSequential(*[SpatialTransformer(cur_ch, num_heads, dim_head, depth=1, context_dim=out_ch)])
                elif connector_config["connector_in_type"] == "additive":
                    axis_to_align_x = 2
                    axis_to_align_c = 2
                    layer = TimestepEmbedSequential(*[AdditiveConditionTransformer(cur_ch, num_heads, dim_head, axis_to_align_x, axis_to_align_c, depth=1, context_dim=out_ch)])
            else:
                layer = None
            connector_in_up.append(layer)

        return nn.ModuleList([nn.ModuleList(connector_in_down), nn.ModuleList(connector_in_up)])
    

    def _setup_connector_out(self, unet_config, connector_config, is_video=False):
        connector_out = []
        in_ch = unet_config.in_channels
        if connector_config["connector_out_input"] == "x_pred":
            in_ch *= 2
        out_ch = connector_config["connector_channel"]
        model_ch = unet_config.block_out_channels[0]
        cur_ch = model_ch // 2
        t_dim = unet_config.time_embedding_dim if "time_embedding_dim" in unet_config and unet_config.time_embedding_dim is not None else unet_config.block_out_channels[0] * 4
        # if "class_embeddings_concat" in unet_config and unet_config.class_embeddings_concat:
        #     t_dim = 2 * t_dim
        video_dim_scale_factor = 4
        ResBlockPreset = partial(ResBlock, dropout=0, dims=2, use_scale_shift_norm=False)
        connector_out.append(TimestepEmbedSequential(nn.Conv2d(in_ch, cur_ch, 3, padding=1, bias=True)))
        for i, mult in enumerate(connector_config["channel_mult_connector"]):
            for _ in range(connector_config["num_blocks_connector"][i]):
                if is_video:
                    layer = [nn.ModuleList([ResBlockPreset(cur_ch, t_dim, out_channels = mult * model_ch),
                            SpatioTemporalAttention(dim = mult * model_ch, dim_head = mult * model_ch // video_dim_scale_factor, heads = 8)])]
                else:
                    layer = [ResBlockPreset(cur_ch, t_dim, out_channels = mult * model_ch)]
                cur_ch = mult * model_ch
                connector_out.append(TimestepEmbedSequential(*layer))
            if i != len(connector_config["channel_mult_connector"]) - 1:
                connector_out.append(TimestepEmbedSequential(Downsample(cur_ch, use_conv=True, dims=2, out_channels=cur_ch)))        
        out = TimestepEmbedSequential(*[normalization(cur_ch), nn.SiLU(), nn.Conv2d(cur_ch, out_ch, 3, padding=1)])
        connector_out.append(out)
        return nn.ModuleList(connector_out)


    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value


    def forward(self, unets, x, timesteps, context, class_labels, xtype=["audio", "video"], x_con=[None, None], t_con=[None, None], block_connection=None):
        # x: list of torch.Tensor
        # timesteps: list of torch.Tensor
        # context: list of torch.Tensor
        # class_labels: list of torch.Tensor
        # xtype: list of str
        
        # Prepare conditioning
        for i in range(len(context)):
            if xtype[i] == "audio" and unets[i].encoder_hid_proj is not None:
                context[i] = unets[i].encoder_hid_proj(context[i])
                
        # Prepare inputs
        bs = x[0].shape[0]
        x = [temp.cuda() if temp is not None else None for temp in x]
        timesteps = [temp.cuda().expand(bs) if temp is not None else None for temp in timesteps]
        x_con = [temp.cuda() if temp is not None else None for temp in x_con]
        t_con = [temp.cuda().expand(bs) if temp is not None else None for temp in t_con]
        context = [temp.cuda() if temp is not None else None for temp in context]
        emb = []
        emb_con = []
        for i, xtype_i in enumerate(xtype):
            if xtype_i == "video":
                t_emb_video = unets[i].time_proj(timesteps[i]).to(dtype=unets[i].dtype)
                emb_video = unets[i].time_embedding(t_emb_video)
                if t_con[i] is None:
                    emb_con.append(emb_video) # do not use class labels for connectors
                else:
                    t_emb_video_con = unets[i].time_proj(t_con[i]).to(dtype=unets[i].dtype)
                    emb_video_con = unets[i].time_embedding(t_emb_video_con)
                    emb_con.append(emb_video_con) # do not use class labels for connectors
                if class_labels[i] is not None and unets[i].class_embedding is not None:
                    class_emb = unets[i].class_embedding(class_labels[i])
                    emb_video = emb_video + class_emb
                    if unets[i].config.class_embeddings_concat:
                        emb_video = torch.cat([emb_video, class_emb], dim=-1)
                    else:
                        emb_video = emb_video + class_emb
                emb.append(emb_video)
            if xtype_i == "audio":    
                t_emb_audio = unets[i].time_proj(timesteps[i]).to(dtype=unets[i].dtype)
                emb_audio = unets[i].time_embedding(t_emb_audio)
                if unets[i].time_embed_act is not None:
                    emb_audio = unets[i].time_embed_act(emb_audio)
                if t_con[i] is None:
                    emb_con.append(emb_audio) # do not use class labels for connectors
                else:
                    t_emb_audio_con = unets[i].time_proj(t_con[i]).to(dtype=unets[i].dtype)
                    emb_audio_con = unets[i].time_embedding(t_emb_audio_con)
                    if unets[i].time_embed_act is not None:
                        emb_audio_con = unets[i].time_embed_act(emb_audio_con)
                    emb_con.append(emb_audio_con) # do not use class labels for connectors
                if class_labels[i] is not None and unets[i].class_embedding is not None:
                    class_emb = unets[i].class_embedding(class_labels[i])
                    if unets[i].config.class_embeddings_concat:
                        emb_audio = torch.cat([emb_audio, class_emb], dim=-1)
                    else:
                        emb_audio = emb_audio + class_emb
                emb.append(emb_audio)

        # Environment encoders (called connectors in this implementation)
        if len(xtype) > 1: # this means two outputs present and thus joint decoding (connectors are required only in this case)
            h_con_orig = [temp for temp in x]
            for i in range(len(xtype)):
                if x_con[i] is not None:
                    h_con_orig[i] = x_con[i]

                c_out = None
                if xtype[i] == "video":
                    c_out = self.connector_out_video
                    c_in_config = self.connector_video_config
                elif xtype[i] == "audio":
                    c_out = self.connector_out_audio
                    c_in_config = self.connector_audio_config
                else:
                    raise

                for j in range(len(c_out)):
                    h_con_orig[i] = c_out[j](h_con_orig[i], emb_con[i], context[i])

                # convert the output of c_out
                if c_in_config["connector_in_type"] == "additive":
                    if h_con_orig[i].ndim == 5:
                        h_con_orig[i] = h_con_orig[i].mean(3).mean(3)
                    else:
                        h_con_orig[i] = h_con_orig[i].mean(3)
                    h_con_orig[i] = h_con_orig[i] / torch.norm(h_con_orig[i], dim=-1, keepdim=True)
                    h_con_orig[i] = h_con_orig[i].unsqueeze(-1)
                else:
                    # codi style (a single vector)
                    if h_con_orig[i].ndim == 5:
                        h_con_orig[i] = h_con_orig[i].mean(2).mean(2).mean(2).unsqueeze(1)
                    else:
                        h_con_orig[i] = h_con_orig[i].mean(2).mean(2).unsqueeze(1)
                    h_con_orig[i] = h_con_orig[i] / torch.norm(h_con_orig[i], dim=-1, keepdim=True)
            
            h_con = []
            for i in range(len(h_con_orig)):
                h_con.append(h_con_orig[1-i])

            block_connection = [block_connection,] * len(xtype)
            if block_connection[0] is not None:
                for i in range(len(xtype)):
                    for j in range(3):
                        block_connection[i] = block_connection[i].unsqueeze(-1)
                    if xtype[i] == "video":
                        if c_in_config["connector_in_type"] == "additive":
                            block_connection[i] = block_connection[i].unsqueeze(-1)
                        else:
                            block_connection[i] = block_connection[i].repeat_interleave(repeats=x[i].shape[2], dim=0)
        else:
            h_con = None
        
        # Joint / single generation
        out_all = []
        for i, xtype_i in enumerate(xtype):
            unet_cur = unets[i]
            if xtype_i == "video":
                c_in = self.connector_in_video
                c_in_config = self.connector_video_config
            elif xtype_i == "audio":
                c_in = self.connector_in_audio
                c_in_config = self.connector_audio_config
            else:
                raise

            # preprocess at unet
            h = x[i]
            if xtype_i == "video":
                # only required for unet_motion_models
                num_frames = h.shape[2]
                h = h.permute(0, 2, 1, 3, 4).reshape((h.shape[0] * num_frames, -1) + h.shape[3:])
                emb_interleaved = [temp.repeat_interleave(repeats=num_frames, dim=0) for temp in emb]
                context_interleaved = [temp.repeat_interleave(repeats=num_frames, dim=0) if temp is not None else None for temp in context]
                if h_con is not None and c_in_config["connector_in_type"] != "additive":
                    h_con[i] = h_con[i].repeat_interleave(repeats=num_frames, dim=0)
            h = unet_cur.conv_in(h)
            if isinstance(unet_cur, UNet3DConditionModel):
                h = unet_cur.transformer_in(h, num_frames=num_frames, return_dict=False)[0]

            # down blocks
            h_res_samples = (h,)
            for j, down_sample_block in enumerate(unet_cur.down_blocks):
                input_args = {}
                has_cross_attention = hasattr(down_sample_block, "has_cross_attention") and down_sample_block.has_cross_attention
                if xtype_i == "video":
                    input_args["temb"] = emb_interleaved[i]
                    if has_cross_attention:
                        input_args["encoder_hidden_states"] = context_interleaved[i]
                    input_args["num_frames"] = num_frames
                else:
                    input_args["temb"] = emb[i]
                    if has_cross_attention:
                        input_args["encoder_hidden_states"] = context[i]
                
                # down blocks in unet
                h, h_res = down_sample_block(h, **input_args)

                # connector
                if h_con is not None and c_in[0][j] is not None:
                    if xtype_i == "video" and c_in_config["connector_in_type"] == "additive":
                        h = rearrange(h, "(b t) c h w -> b c t h w", t=num_frames)
                    h_new = c_in[0][j](h, context=h_con[i])
                    if block_connection[i] is None:
                        h = h_new
                    else:
                        h = torch.where(block_connection[i].to(h.device), h, h_new)
                    if xtype_i == "video" and c_in_config["connector_in_type"] == "additive":
                        h = rearrange(h, "b c t h w -> (b t) c h w")
                    h_res = h_res[:-1]
                    h_res += (h,)

                h_res_samples += h_res

            # middle blocks
            if xtype_i == "video":
                h = unet_cur.mid_block(h, emb_interleaved[i], context_interleaved[i], num_frames=num_frames)
            else:
                h = unet_cur.mid_block(h, emb[i], context[i])

            # up blocks
            for j, up_sample_block in enumerate(unet_cur.up_blocks):
                input_args = {}
                has_cross_attention = hasattr(up_sample_block, "has_cross_attention") and up_sample_block.has_cross_attention
                if xtype_i == "video":
                    input_args["temb"] = emb_interleaved[i]
                    if has_cross_attention:
                        input_args["encoder_hidden_states"] = context_interleaved[i]
                    input_args["num_frames"] = num_frames
                else:
                    input_args["temb"] = emb[i]
                    if has_cross_attention:
                        input_args["encoder_hidden_states"] = context[i]

                # up blocks in unet
                h_res = h_res_samples[-len(up_sample_block.resnets):]
                h_res_samples = h_res_samples[:-len(up_sample_block.resnets)]
                h = up_sample_block(h, res_hidden_states_tuple=h_res, **input_args)

                # connector
                if h_con is not None and c_in[1][j] is not None:
                    if xtype_i == "video" and c_in_config["connector_in_type"] == "additive":
                        h = rearrange(h, "(b t) c h w -> b c t h w", t=num_frames)
                    h_new = c_in[1][j](h, context=h_con[i])
                    if block_connection[i] is None:
                        h = h_new
                    else:
                        h = torch.where(block_connection[i].to(h.device), h, h_new)
                    if xtype_i == "video" and c_in_config["connector_in_type"] == "additive":
                        h = rearrange(h, "b c t h w -> (b t) c h w")
                
            # post-process at unet
            if unet_cur.conv_norm_out:
                h = unet_cur.conv_norm_out(h)
                h = unet_cur.conv_act(h)
            h = unet_cur.conv_out(h)

            if xtype_i == "video":
                # only required for unet_motion_models
                h = h[None, :].reshape((-1, num_frames) + h.shape[1:]).permute(0, 2, 1, 3, 4)

            out_all.append(h)

        return out_all
    