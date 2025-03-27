from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
# from diffusers.models.activations import get_activation
# from diffusers.loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
from dataclasses import dataclass
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
# from diffusers.models.attention_processor import (
#     ADDED_KV_ATTENTION_PROCESSORS,
#     CROSS_ATTENTION_PROCESSORS,
#     Attention,
#     AttentionProcessor,
#     AttnAddedKVProcessor,
#     AttnProcessor,
# )
# from diffusers.models.embeddings import (
#     GaussianFourierProjection,
#     GLIGENTextBoundingboxProjection,
#     ImageHintTimeEmbedding,
#     ImageProjection,
#     ImageTimeEmbedding,
#     TextImageProjection,
#     TextImageTimeEmbedding,
#     TextTimeEmbedding,
#     TimestepEmbedding,
#     Timesteps,
# )
from diffusers.models.modeling_utils import ModelMixin
# from diffusers.models.unet_2d_blocks import (
#     UNetMidBlock2D,
#     UNetMidBlock2DCrossAttn,
#     UNetMidBlock2DSimpleCrossAttn,
#     get_down_block,
#     get_up_block,
# )
import abc

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            if self.activate:
                self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        

class dict2obj(object):
    def __init__(self, d):
        self.__dict__['d'] = d
 
    def __getattr__(self, key):
        value = self.__dict__['d'][key]
        if type(value) == type({}):
            return dict2obj(value)
 
        return value


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if self.activate:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
    #         if attn.shape[1] <= 128 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if self.activate:
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
            self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.activate = True



def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            
            
            attn_1 = controller(attn, is_cross, place_in_unet)
            
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            
            out = self.reshape_batch_dim_to_heads(out)
            out = self.to_out(out)
            return out

        return forward

    def register_recr(net_, count, place_in_unet):
#         print(net_.__class__.__name__,place_in_unet)
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
#     sub_nets = model.unet.named_children()
    sub_nets = model.named_children()
    for net in sub_nets:
#         print(net[0])
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count


@dataclass
class CustomUNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None
    multi_level_feats: [torch.FloatTensor] = None
    sample_320: torch.FloatTensor = None

class CustomUNet2DConditionModel(UNet2DConditionModel):

    # def __init__(
    #     self, 
    #     use_attn=False, 
    #     base_size=512, 
    #     max_attn_size=None, 
    #     attn_selector='up_cross+down_cross',
    #     kwargs):

    #     super.__init__(kwargs)

    #     self.attention_store = AttentionStore(base_size=base_size // 8, max_size=max_attn_size)
    #     self.size16 = base_size // 32
    #     self.size32 = base_size // 16
    #     self.size64 = base_size // 8
    #     self.use_attn = use_attn
    #     if self.use_attn:
    #         register_attention_control(unet, self.attention_store)
    #     register_hier_output(unet)
    #     self.unet = unet
    #     self.attn_selector = attn_selector.split('+')

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
        ) -> Union[UNet2DConditionOutput, Tuple]:
            r"""
            The [`UNet2DConditionModel`] forward method.

            Args:
                sample (`torch.FloatTensor`):
                    The noisy input tensor with the following shape `(batch, channel, height, width)`.
                timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
                encoder_hidden_states (`torch.FloatTensor`):
                    The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
                class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                    Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
                timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                    Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                    through the `self.time_embedding` layer to obtain the timestep embeddings.
                attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                    An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                    is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                    negative values to the attention scores corresponding to "discard" tokens.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                    `self.processor` in
                    [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                added_cond_kwargs: (`dict`, *optional*):
                    A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                    are passed along to the UNet blocks.
                down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                    A tuple of tensors that if specified are added to the residuals of down unet blocks.
                mid_block_additional_residual: (`torch.Tensor`, *optional*):
                    A tensor that if specified is added to the residual of the middle unet block.
                encoder_attention_mask (`torch.Tensor`):
                    A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                    `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                    which adds large negative values to the attention scores corresponding to "discard" tokens.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                    tuple.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the [`AttnProcessor`].
                added_cond_kwargs: (`dict`, *optional*):
                    A kwargs dictionary containin additional embeddings that if specified are added to the embeddings that
                    are passed along to the UNet blocks.
                down_block_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                    additional residuals to be added to UNet long skip connections from down blocks to up blocks for
                    example from ControlNet side model(s)
                mid_block_additional_residual (`torch.Tensor`, *optional*):
                    additional residual to be added to UNet mid block output, for example from ControlNet side model
                down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                    additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)

            Returns:
                [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                    If `return_dict` is True, an [`~models.unet_2d_condition.UNet2DConditionOutput`] is returned, otherwise
                    a `tuple` is returned where the first element is the sample tensor.
            """
            # By default samples have to be AT least a multiple of the overall upsampling factor.
            # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
            # However, the upsampling interpolation output size can be forced to fit any upsampling size
            # on the fly if necessary.
            default_overall_up_factor = 2**self.num_upsamplers

            # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
            forward_upsample_size = False
            upsample_size = None

            for dim in sample.shape[-2:]:
                if dim % default_overall_up_factor != 0:
                    # Forward upsample size to force interpolation output size.
                    forward_upsample_size = True
                    break

            # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
            # expects mask of shape:
            #   [batch, key_tokens]
            # adds singleton query_tokens dimension:
            #   [batch,                    1, key_tokens]
            # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
            #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
            #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
            if attention_mask is not None:
                # assume that mask is expressed as:
                #   (1 = keep,      0 = discard)
                # convert mask into a bias that can be added to attention scores:
                #       (keep = +0,     discard = -10000.0)
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # convert encoder_attention_mask to a bias the same way we do for attention_mask
            if encoder_attention_mask is not None:
                encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

            # 0. center input if necessary
            if self.config.center_input_sample:
                sample = 2 * sample - 1.0

            # 1. time
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(sample.shape[0])

            t_emb = self.time_proj(timesteps)

            # `Timesteps` does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=sample.dtype)

            emb = self.time_embedding(t_emb, timestep_cond)
            aug_emb = None

            if self.class_embedding is not None:
                if class_labels is None:
                    raise ValueError("class_labels should be provided when num_class_embeds > 0")

                if self.config.class_embed_type == "timestep":
                    class_labels = self.time_proj(class_labels)

                    # `Timesteps` does not contain any weights and will always return f32 tensors
                    # there might be better ways to encapsulate this.
                    class_labels = class_labels.to(dtype=sample.dtype)

                class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

                if self.config.class_embeddings_concat:
                    emb = torch.cat([emb, class_emb], dim=-1)
                else:
                    emb = emb + class_emb

            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)
            elif self.config.addition_embed_type == "text_image":
                # Kandinsky 2.1 - style
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                    )

                image_embs = added_cond_kwargs.get("image_embeds")
                text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
                aug_emb = self.add_embedding(text_embs, image_embs)
            elif self.config.addition_embed_type == "text_time":
                # SDXL - style
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                    )
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)
            elif self.config.addition_embed_type == "image":
                # Kandinsky 2.2 - style
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                    )
                image_embs = added_cond_kwargs.get("image_embeds")
                aug_emb = self.add_embedding(image_embs)
            elif self.config.addition_embed_type == "image_hint":
                # Kandinsky 2.2 - style
                if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                    )
                image_embs = added_cond_kwargs.get("image_embeds")
                hint = added_cond_kwargs.get("hint")
                aug_emb, hint = self.add_embedding(image_embs, hint)
                sample = torch.cat([sample, hint], dim=1)

            emb = emb + aug_emb if aug_emb is not None else emb

            if self.time_embed_act is not None:
                emb = self.time_embed_act(emb)

            if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
                encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
            elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
                # Kadinsky 2.1 - style
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                    )

                image_embeds = added_cond_kwargs.get("image_embeds")
                encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
            elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
                # Kandinsky 2.2 - style
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                    )
                image_embeds = added_cond_kwargs.get("image_embeds")
                encoder_hidden_states = self.encoder_hid_proj(image_embeds)
            elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                    )
                image_embeds = added_cond_kwargs.get("image_embeds")
                image_embeds = self.encoder_hid_proj(image_embeds).to(encoder_hidden_states.dtype)
                encoder_hidden_states = torch.cat([encoder_hidden_states, image_embeds], dim=1)

            # 2. pre-process
            sample = self.conv_in(sample)

            # 2.5 GLIGEN position net
            if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
                cross_attention_kwargs = cross_attention_kwargs.copy()
                gligen_args = cross_attention_kwargs.pop("gligen")
                cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

            # 3. down
            lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
            if USE_PEFT_BACKEND:
                # weight the lora layers by setting `lora_scale` for each PEFT layer
                scale_lora_layers(self, lora_scale)

            is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
            # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
            is_adapter = down_intrablock_additional_residuals is not None
            # maintain backward compatibility for legacy usage, where
            #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
            #       but can only use one or the other
            if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
                deprecate(
                    "T2I should not use down_block_additional_residuals",
                    "1.3.0",
                    "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                        and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                        for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                    standard_warn=False,
                )
                down_intrablock_additional_residuals = down_block_additional_residuals
                is_adapter = True

            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    # For t2i-adapter CrossAttnDownBlock2D
                    additional_residuals = {}
                    if is_adapter and len(down_intrablock_additional_residuals) > 0:
                        additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                        **additional_residuals,
                    )
                else:
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=lora_scale)
                    if is_adapter and len(down_intrablock_additional_residuals) > 0:
                        sample += down_intrablock_additional_residuals.pop(0)

                down_block_res_samples += res_samples

            if is_controlnet:
                new_down_block_res_samples = ()

                for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
                ):
                    down_block_res_sample = down_block_res_sample + down_block_additional_residual
                    new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

                down_block_res_samples = new_down_block_res_samples

            # 4. mid
            if self.mid_block is not None:
                if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = self.mid_block(sample, emb)

                # To support T2I-Adapter-XL
                if (
                    is_adapter
                    and len(down_intrablock_additional_residuals) > 0
                    and sample.shape == down_intrablock_additional_residuals[0].shape
                ):
                    sample += down_intrablock_additional_residuals.pop(0)

            if is_controlnet:
                sample = sample + mid_block_additional_residual

            multi_level_feats = []
           # 1, 1280, 24, 24
            multi_level_feats.append(sample) # 1/64
            # 5. up
            for i, upsample_block in enumerate(self.up_blocks):
                is_final_block = i == len(self.up_blocks) - 1

                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                        scale=lora_scale,
                    )
                if not is_final_block:
                    multi_level_feats.append(sample)

            # 6. post-process
            if self.conv_norm_out:
                sample = self.conv_norm_out(sample)
                sample = self.conv_act(sample)
            
            # sample_320 = sample


            sample = self.conv_out(sample)

            if USE_PEFT_BACKEND:
                # remove `lora_scale` from each PEFT layer
                unscale_lora_layers(self, lora_scale)

            if not return_dict:
                return (sample,)

            return CustomUNet2DConditionOutput(
                sample=sample, 
                multi_level_feats=multi_level_feats,
                # sample_320=sample_320
            )

class UNetWrapper(nn.Module):
    def __init__(self, unet, use_attn=False, base_size=512, max_attn_size=None, attn_selector='up_cross+down_cross') -> None:
        super().__init__()
        self.attention_store = AttentionStore(base_size=base_size // 8, max_size=max_attn_size)
        self.size16 = base_size // 32
        self.size32 = base_size // 16
        self.size64 = base_size // 8
        self.use_attn = use_attn
        if self.use_attn:
            register_attention_control(unet, self.attention_store)
        register_hier_output(unet)
        self.unet = unet
        self.attn_selector = attn_selector.split('+')

    def forward(self, latents, t, c_crossattn):
        if self.use_attn:
            self.attention_store.reset()
        out_list = self.unet(x=latents, t=t, c_crossattn=c_crossattn)
        if self.use_attn:
            avg_attn = self.attention_store.get_average_attention()
            attn16, attn32, attn64 = self.process_attn(avg_attn)
            out_list[1] = torch.cat([out_list[1], attn16], dim=1)
            out_list[2] = torch.cat([out_list[2], attn32], dim=1)
            if attn64 is not None:
                out_list[3] = torch.cat([out_list[3], attn64], dim=1)

        return out_list[::-1]

    def process_attn(self, avg_attn):
        attns = {self.size16: [], self.size32: [], self.size64: []}
        for k in self.attn_selector:
            for up_attn in avg_attn[k]:
                size = int(math.sqrt(up_attn.shape[1]))
                attns[size].append(rearrange(up_attn, 'b (h w) c -> b c h w', h=size))
        attn16 = torch.stack(attns[self.size16]).mean(0)
        attn32 = torch.stack(attns[self.size32]).mean(0)
        if len(attns[self.size64]) > 0:
            attn64 = torch.stack(attns[self.size64]).mean(0)
        else:
            attn64 = None
        return attn16, attn32, attn64    