# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

import copy
from tqdm.auto import trange
import random
from PIL import Image

# pulid imports
import torch.nn as nn
import insightface
import gc
import cv2
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize
from huggingface_hub import hf_hub_download, snapshot_download
from insightface.app import FaceAnalysis
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import sys

sys.path.append('../PuLID')

from pulid.encoders_flux import IDFormer, PerceiverAttentionCA
from pulid.utils import img2tensor, tensor2img, resize_numpy_image_long
from eva_clip import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxImg2ImgPipeline
        >>> from diffusers.utils import load_image

        >>> device = "cuda"
        >>> pipe = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe = pipe.to(device)

        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
        >>> init_image = load_image(url).resize((1024, 1024))

        >>> prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

        >>> images = pipe(
        ...     prompt=prompt, image=init_image, num_inference_steps=4, strength=0.95, guidance_scale=0.0
        ... ).images[0]
        ```
"""

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class RegionalFluxAttnProcessor2_0:
    def __init__(self):  
        self.regional_mask = None
    def FluxAttnProcessor2_0_call(
        self,
        attn,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask = None,
        image_rotary_emb = None,
    ) -> torch.FloatTensor:
        
        batch_size, _, _ = hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            
            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
            
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # apply mask on attention
        hidden_states = torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
            
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
    
    def __call__(
        self,
        attn,
        hidden_states,
        hidden_states_base = None,
        encoder_hidden_states = None,
        encoder_hidden_states_base = None,
        attention_mask = None,
        image_rotary_emb = None,
        image_rotary_emb_base = None,
        additional_kwargs = None,
        base_ratio = None,
    ) -> torch.FloatTensor:
        
        if base_ratio is not None:
            attn_output_base = self.FluxAttnProcessor2_0_call(
                attn=attn,
                hidden_states=hidden_states_base if hidden_states_base is not None else hidden_states,
                encoder_hidden_states=encoder_hidden_states_base,
                attention_mask=None,
                image_rotary_emb=image_rotary_emb_base,
            )

            if encoder_hidden_states_base is not None:
                hidden_states_base, encoder_hidden_states_base = attn_output_base
            else:
                hidden_states_base = attn_output_base

        # move regional mask to device
        if base_ratio is not None and 'regional_attention_mask' in additional_kwargs:
            if self.regional_mask is not None:
                regional_mask = self.regional_mask.to(hidden_states.device)
            else:
                self.regional_mask = additional_kwargs['regional_attention_mask']
                regional_mask = self.regional_mask.to(hidden_states.device)
        else:
            regional_mask = None

        attn_output = self.FluxAttnProcessor2_0_call(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=regional_mask,
            image_rotary_emb=image_rotary_emb,
        )

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = attn_output
        else:
            hidden_states = attn_output
        
        if encoder_hidden_states is not None:

            if base_ratio is not None:
                # merge hidden_states and hidden_states_base
                hidden_states = hidden_states*(1-base_ratio) + hidden_states_base*base_ratio
                return hidden_states, encoder_hidden_states, encoder_hidden_states_base
            else: # both regional and base input are base prompts, skip the merge
                return hidden_states, encoder_hidden_states, encoder_hidden_states
        
        else:
            if base_ratio is not None:
                
                encoder_hidden_states, hidden_states = (
                    hidden_states[:, : additional_kwargs['encoder_seq_len']],
                    hidden_states[:, additional_kwargs['encoder_seq_len'] :],
                )
               
                encoder_hidden_states_base, hidden_states_base = (
                    hidden_states_base[:, : additional_kwargs["encoder_seq_len_base"]],
                    hidden_states_base[:, additional_kwargs["encoder_seq_len_base"] :],
                )

                # merge hidden_states and hidden_states_base
                hidden_states = hidden_states*(1-base_ratio) + hidden_states_base*base_ratio

                # concat back            
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                hidden_states_base = torch.cat([encoder_hidden_states_base, hidden_states_base], dim=1)
    
                return hidden_states, hidden_states_base

            else: # both regional and base input are base prompts, skip the merge
                return hidden_states, hidden_states
        

class RegionalFluxPipeline_PULID(FluxPipeline):

    def load_pulid_models(self):
        double_interval = 2
        single_interval = 4
        num_ca = 0
        onnx_provider = 'gpu'

        # init encoder
        self.pulid_encoder = IDFormer().to(self.device, self.transformer.dtype)

        num_ca = 19 // double_interval + 38 // single_interval
        if 19 % double_interval != 0:
            num_ca += 1
        if 38 % single_interval != 0:
            num_ca += 1
        self.pulid_ca = nn.ModuleList([
            PerceiverAttentionCA().to(self.device, self.transformer.dtype) for _ in range(num_ca)
        ])

        self.transformer.pulid_ca = self.pulid_ca
        self.transformer.pulid_double_interval = double_interval
        self.transformer.pulid_single_interval = single_interval

        # preprocessors
        # face align and parsing
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device)
        # clip-vit backbone
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        model = model.visual
        self.clip_vision_model = model.to(self.device, dtype=self.transformer.dtype)
        eva_transform_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std
        # antelopev2
        snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
        providers = ['CPUExecutionProvider'] if onnx_provider == 'cpu' \
            else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.app = FaceAnalysis(name='antelopev2', root='.', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx',
                                                            providers=providers)
        self.handler_ante.prepare(ctx_id=0)

        gc.collect()
        torch.cuda.empty_cache()

        # self.load_pretrain()

        # other configs
        self.debug_img_list = []
    
    def load_pretrain(self, pretrain_path=None):
        hf_hub_download('guozinan/PuLID', 'pulid_flux_v0.9.1.safetensors', local_dir='models')
        ckpt_path = 'models/pulid_flux_v0.9.1.safetensors'
        if pretrain_path is not None:
            ckpt_path = pretrain_path
        state_dict = load_file(ckpt_path)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1:]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            print(f'loading from {module}')
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

        del state_dict
        del state_dict_dict

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    @torch.no_grad()
    def get_id_embedding(self, image, cal_uncond=False):
        """
        Args:
            image: path
        """
        image = np.array(Image.open(image))
        image = resize_numpy_image_long(image, 1024)

        self.face_helper.clean_all()
        self.debug_img_list = []
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get antelopev2 embedding
        face_info = self.app.get(image_bgr)
        if len(face_info) > 0:
            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
                -1
            ]  # only use the maximum face
            id_ante_embedding = face_info['embedding']
            self.debug_img_list.append(
                image[
                    int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                    int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                ]
            )
        else:
            id_ante_embedding = None

        # using facexlib to detect and align face
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib align face fail')
        align_face = self.face_helper.cropped_faces[0]
        # incase insightface didn't detect face
        if id_ante_embedding is None:
            print('fail to detect face using insightface, extract embedding on align face')
            id_ante_embedding = self.handler_ante.get_feat(align_face)

        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device, self.transformer.dtype)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        # parsing
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(self.device)
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, self.to_gray(input))
        self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

        # transform img before sending to eva-clip-vit
        face_features_image = resize(face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)
        face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
        id_cond_vit, id_vit_hidden = self.clip_vision_model(
            face_features_image.to(self.transformer.dtype), return_all_features=False, return_hidden=True, shuffle=False
        )
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

        id_embedding = self.pulid_encoder(id_cond, id_vit_hidden)

        if not cal_uncond:
            return id_embedding, None

        id_uncond = torch.zeros_like(id_cond)
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(id_vit_hidden)):
            id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[layer_idx]))
        uncond_id_embedding = self.pulid_encoder(id_uncond, id_vit_hidden_uncond)

        return id_embedding, uncond_id_embedding

    @torch.inference_mode()
    def __call__(
            self,
            initial_latent: torch.FloatTensor = None,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            num_samples: int = 1,
            width: int = 1024,
            height: int = 1024,
            strength: float = 1.0,
            num_inference_steps: int = 25,
            timesteps: List[int] = None,
            mask_inject_steps: int = 5,
            guidance_scale: float = 5.0,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self._guidance_scale = guidance_scale

        device = self.transformer.device

        # 3. Define call parameters
        batch_size = num_samples if num_samples else prompt_embeds.shape[0]
        
        # encode base prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=512,
            lora_scale=None,
        )

        # define base mask and inputs
        base_mask = torch.ones((height, width), device=device, dtype=self.transformer.dtype) # base mask uses the whole image mask
        base_inputs = [(base_mask, prompt_embeds)]

        # encode regional prompts, define regional inputs
        regional_inputs = []
        if 'regional_prompts' in joint_attention_kwargs and 'regional_masks' in joint_attention_kwargs:
            for regional_prompt, regional_mask in zip(joint_attention_kwargs['regional_prompts'], joint_attention_kwargs['regional_masks']):
                regional_prompt_embeds, regional_pooled_prompt_embeds, regional_text_ids = self.encode_prompt(
                    prompt=regional_prompt,
                    prompt_2=regional_prompt,
                    prompt_embeds=None,
                    pooled_prompt_embeds=None,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=512,
                    lora_scale=None,
                )
                regional_inputs.append((regional_mask, regional_prompt_embeds))
        
        ## prepare masks for regional control
        conds = []
        masks = []
        H, W = height//(self.vae_scale_factor), width//(self.vae_scale_factor)
        hidden_seq_len = H * W
        for mask, cond in regional_inputs:
            if mask is not None: # resize regional masks to image size, the flatten is to match the seq len
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], (H, W), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, cond.size(1))
            else:
                mask = torch.ones((H*W, cond.size(1))).to(device=cond.device)
            masks.append(mask)
            conds.append(cond)
        regional_embeds = torch.cat(conds, dim=1)
        encoder_seq_len = regional_embeds.shape[1]

        # initialize attention mask
        regional_attention_mask = torch.zeros(
            (encoder_seq_len + hidden_seq_len, encoder_seq_len + hidden_seq_len),
            device=masks[0].device,
            dtype=torch.bool
        )
        num_of_regions = len(masks)
        each_prompt_seq_len = encoder_seq_len // num_of_regions

        # initialize self-attended mask
        self_attend_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

        # initialize union mask
        union_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

        # handle each mask
        for i in range(num_of_regions):
            # txt attends to itself
            regional_attention_mask[i*each_prompt_seq_len:(i+1)*each_prompt_seq_len, i*each_prompt_seq_len:(i+1)*each_prompt_seq_len] = True

            # txt attends to corresponding regional img
            regional_attention_mask[i*each_prompt_seq_len:(i+1)*each_prompt_seq_len, encoder_seq_len:] = masks[i].transpose(-1, -2)

            # regional img attends to corresponding txt
            regional_attention_mask[encoder_seq_len:, i*each_prompt_seq_len:(i+1)*each_prompt_seq_len] = masks[i]

            # regional img attends to corresponding regional img
            img_size_masks = masks[i][:, :1].repeat(1, hidden_seq_len)
            img_size_masks_transpose = img_size_masks.transpose(-1, -2)
            self_attend_masks = torch.logical_or(self_attend_masks, 
                                                    torch.logical_and(img_size_masks, img_size_masks_transpose))

            # update union
            union_masks = torch.logical_or(union_masks, 
                                            torch.logical_or(img_size_masks, img_size_masks_transpose))

        background_masks = torch.logical_not(union_masks)

        background_and_self_attend_masks = torch.logical_or(background_masks, self_attend_masks)

        regional_attention_mask[encoder_seq_len:, encoder_seq_len:] = background_and_self_attend_masks
        ## done prepare masks for regional control

        ## prepare id embeddings
        if 'id_image_paths' in joint_attention_kwargs:
            id_embeddings = []
            for id_image_path in joint_attention_kwargs['id_image_paths']:
                id_embedding, _ = self.get_id_embedding(id_image_path, cal_uncond=False)
                id_embeddings.append(id_embedding)
            id_masks = []
            for id_mask in joint_attention_kwargs['id_masks']:
                id_mask = torch.nn.functional.interpolate(id_mask[None, None, :, :], (H, W), mode='nearest-exact').flatten()
                id_masks.append(id_mask)
        else:
            id_embeddings = None
            id_masks = None

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4 
        latents, latent_image_ids = self.prepare_latents( 
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.transformer.dtype,
            device,
            generator,
            initial_latent,
        )
    
        # 4.Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        
        image_seq_len = (int(height) // self.vae_scale_factor) * (int(width) // self.vae_scale_factor)
        
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps, 
            sigmas,
            mu=mu,
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

       # 5.handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                if i < mask_inject_steps:
                    chosen_prompt_embeds = regional_embeds
                    base_ratio = joint_attention_kwargs['base_ratio']
                else:
                    chosen_prompt_embeds = prompt_embeds
                    base_ratio = None
                
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=chosen_prompt_embeds,
                    encoder_hidden_states_base=prompt_embeds,
                    base_ratio=base_ratio,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs={
                        'single_inject_blocks_interval': joint_attention_kwargs['single_inject_blocks_interval'] if 'single_inject_blocks_interval' in joint_attention_kwargs else len(self.transformer.single_transformer_blocks), 
                        'double_inject_blocks_interval': joint_attention_kwargs['double_inject_blocks_interval'] if 'double_inject_blocks_interval' in joint_attention_kwargs else len(self.transformer.transformer_blocks),
                        'regional_attention_mask': regional_attention_mask if base_ratio is not None else None,
                        'id_embeddings': id_embeddings,
                        'id_weights': joint_attention_kwargs['id_weights'] if 'id_weights' in joint_attention_kwargs else None,
                        'id_masks': id_masks,
                    },
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)