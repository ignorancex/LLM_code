from dataclasses import dataclass
import torch
import torch.nn as nn
from diffusers import (
    DDPMScheduler,
    StableDiffusionPipeline,
    DDIMScheduler
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers import AutoPipelineForInpainting, StableDiffusionPipeline
import gc
import tinycudann as tcnn
from typing import Any, Dict, List, Optional, Union
from diffusers.image_processor import PipelineImageInput

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()
    
class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x):
        return self.module(x).to(self.dtype)

@dataclass
class Config:
    pretrained_model_name_or_path: str = "./pretrained_models/stable-diffusion-inpainting"
    pretrained_model_name_or_path_lora: str = "./pretrained_models/stable-diffusion-inpainting"
    
    pretrained_model_name_or_path_sd: str = "./pretrained_models/stable-diffusion-v1-5"

    guidance_scale: float = 7.5
    guidance_scale_lora: float = 1.0
    
    half_precision_weights: bool = True
    
class StableDiffusionGuidance:
    
    def __init__(self, blip_rst='', use_lora=False, use_sd15=False, guidance_scale=7.5):
        self.cfg = Config()
        self.blip_rst = blip_rst
        self.use_lora = use_lora
        self.use_sd15 = use_sd15
        self.guidance_scale = guidance_scale

    def configure(self) -> None:
        self.device = torch.device('cuda')
        print("Loading Stable Diffusion Inpainting ...")
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        pipe_kwargs = {
            "safety_checker": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        pipe_lora_kwargs = {
            "safety_checker": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        @dataclass
        class SubModules:
            pipe: AutoPipelineForInpainting
            pipe_lora: AutoPipelineForInpainting
        pipe = AutoPipelineForInpainting.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)
        if self.use_sd15:
            pipe_sd = StableDiffusionPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path_sd,
                **pipe_kwargs,
            ).to(self.device)
            self.pipe_sd = pipe_sd
            del self.pipe_sd.text_encoder
        if (
            self.cfg.pretrained_model_name_or_path
            == self.cfg.pretrained_model_name_or_path_lora
        ):
            self.single_model = True
            pipe_lora = pipe
        else:
            self.single_model = False
            pipe_lora = AutoPipelineForInpainting.from_pretrained(
                self.cfg.pretrained_model_name_or_path_lora,
                **pipe_lora_kwargs,
            ).to(self.device)
            del pipe_lora.vae
            del pipe_lora.text_encoder
            cleanup()
            pipe_lora.vae = pipe.vae
            pipe_lora.text_encoder = pipe.text_encoder
        self.submodules = SubModules(pipe=pipe, pipe_lora=pipe_lora)
        
        cleanup()

        self.pipe.text_encoder.eval()
        self.pipe.vae.eval()
        self.pipe.unet.eval()
        self.pipe_lora.unet.eval()
        
        if self.use_sd15:
            self.pipe_sd.unet.eval()
        
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)
            
        if self.use_sd15:
            for p in self.pipe_sd.unet.parameters():
                p.requires_grad_(False)

        if self.use_lora:
            lora_attn_procs = {}
            for name in self.unet_lora.attn_processors.keys():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else self.unet_lora.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = self.unet_lora.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet_lora.config.block_out_channels[block_id]
                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                )
            self.unet_lora.set_attn_processor(lora_attn_procs)
            self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors).to(
                self.device
            )
            self.lora_layers._load_state_dict_pre_hooks.clear()
            self.lora_layers._state_dict_hooks.clear()
        
        self.scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.scheduler_lora = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path_lora,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        
        if self.use_sd15:
            self.scheduler_sd = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path_sd,
                subfolder='scheduler',
                torch_dtype=self.weights_dtype,
            )

        self.scheduler_sample = DDIMScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.scheduler_lora_sample = DDIMScheduler.from_config(
            self.pipe_lora.scheduler.config
        )
        if self.use_sd15:
            self.scheduelr_sd_sample = DDIMScheduler.from_config(
                self.pipe_sd.scheduler.config
            )

        self.pipe.scheduler = self.scheduler
        self.pipe_lora.scheduler = self.scheduler_lora
        if self.use_sd15:
            self.pipe_sd.scheduler = self.scheduler_sd

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.alphas = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.num_images_per_prompt = 1
        cross_attention_kwargs = {'scale': 1.0}
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        self.cfg.guidance_scale = self.guidance_scale
        prompt_embeds_rst = self.pipe.encode_prompt(
            prompt=self.blip_rst + ', realistic, 8k',
            negative_prompt='blurry, unrealistic',
            device=self.device,
            num_images_per_prompt=self.num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=text_encoder_lora_scale
        )
        self.prompt_embeds = prompt_embeds_rst[0]
        self.negative_prompt_embeds = prompt_embeds_rst[1]
        
        print("Loaded Stable Diffusion Inpainting!")

    @property
    def do_classifier_free_guidance(self):
        return self.cfg.guidance_scale > 1
    
    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def pipe_lora(self):
        return self.submodules.pipe_lora

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def unet_lora(self):
        return self.submodules.pipe_lora.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    @property
    def vae_lora(self):
        return self.submodules.pipe_lora.vae
    
    @property
    def text_encoder(self):
        return self.submodules.pipe.text_encoder

    @property
    def text_encoder_lora(self):
        return self.submodules.pipe_lora.text_encoder

    def cal_warp_sds_grad(
        self,
        image: torch.FloatTensor, # rgb_BCHW: warped_image
        mask_image: torch.FloatTensor, # depth_B1HW: mask_image
        rendered_image: torch.FloatTensor, # rgb_BCHW: rendered_image
        num_inference_steps: int,
        guidance_scale: float, 
        guidance_mask: torch.Tensor = None, 
        num_images_per_prompt: int = 1
    ):
            
        height, width = 512, 512
        
        mask_condition = self.pipe.mask_processor.preprocess(mask_image.to(self.weights_dtype), height=height, width=width, resize_mode='default', crops_coords=None)
        init_warped_image = self.pipe.image_processor.preprocess(image.to(self.weights_dtype), height=height, width=width, crops_coords=None, resize_mode='default')
        masked_image_latents = init_warped_image * (mask_condition < 0.5)
        
        with torch.no_grad():
            warp_sds_grad = self.single_forward_grad(
                self.pipe, 
                t_start=0.25, 
                t_end=0.75, 
                prompt=None, 
                prompt_embeds=self.prompt_embeds, 
                negative_prompt_embeds=None,
                image=rendered_image.to(self.weights_dtype),
                mask_image=mask_condition.to(self.weights_dtype),  
                guidance_scale=guidance_scale,   
                height=height, 
                width=width, 
                masked_image_latents=masked_image_latents.to(self.weights_dtype), 
                cross_attention_kwargs={"scale": 1.0}
            )
        
        warp_sds_grad = torch.nan_to_num(warp_sds_grad)
        init_rendered_image = self.pipe.image_processor.preprocess(rendered_image.to(self.weights_dtype), height=height, width=width, crops_coords=None, resize_mode='default')
        rendered_image_latents = self.pipe._encode_vae_image(image=init_rendered_image.to(self.weights_dtype), generator=None)
        target = (rendered_image_latents - warp_sds_grad).detach()
        if guidance_mask:
            guidance_mask_condition = self.pipe.mask_processor.preprocess(guidance_mask.to(self.weights_dtype), height=height, width=width, resize_mode='default', crops_coords=None)
            guidance_mask_condition = torch.nn.functional.interpolate(
                guidance_mask_condition, size=(height // self.pipe.vae_scale_factor, width // self.pipe.vae_scale_factor)
            )
            rendered_image_latents[guidance_mask_condition] = 0.
            target[guidance_mask_condition] = 0.
        loss_warp_sds = 0.5 * torch.nn.functional.mse_loss(rendered_image_latents.float(), target.float(), reduction='mean')
        
        return loss_warp_sds.to(torch.float32)
    
    def cal_warp_sds_grad_2_2(
        self,
        image: torch.FloatTensor, # rgb_BCHW: warped_image
        mask_image: torch.FloatTensor, # depth_B1HW: mask_image
        rendered_image: torch.FloatTensor, # rgb_BCHW: rendered_image
        num_inference_steps: int,
        guidance_scale: float, 
        guidance_mask: torch.Tensor = None, 
        num_images_per_prompt: int = 1
    ):
            
        height, width = 512, 512
        
        mask_condition = self.pipe.mask_processor.preprocess(mask_image.to(self.weights_dtype), height=height, width=width, resize_mode='default', crops_coords=None)
        init_warped_image = self.pipe.image_processor.preprocess(image.to(self.weights_dtype), height=height, width=width, crops_coords=None, resize_mode='default')
        masked_image_latents = init_warped_image * (mask_condition < 0.5)
        
        with torch.no_grad():
            warp_sds_grad_1, warp_sds_grad_2 = self.single_forward_grad_2(
                self.pipe, 
                self.pipe_sd, 
                t_start=0.25, 
                t_end=0.75, 
                prompt=None, 
                prompt_embeds=self.prompt_embeds, 
                negative_prompt_embeds=None,
                image=rendered_image.to(self.weights_dtype),
                mask_image=mask_condition.to(self.weights_dtype),  
                guidance_scale=guidance_scale,   
                height=height, 
                width=width, 
                masked_image_latents=masked_image_latents.to(self.weights_dtype), 
                cross_attention_kwargs={"scale": 1.0}
            )
        
        warp_sds_grad_1 = torch.nan_to_num(warp_sds_grad_1)
        init_rendered_image = self.pipe.image_processor.preprocess(rendered_image.to(self.weights_dtype), height=height, width=width, crops_coords=None, resize_mode='default')
        rendered_image_latents = self.pipe._encode_vae_image(image=init_rendered_image.to(self.weights_dtype), generator=None)
        target = (rendered_image_latents - warp_sds_grad_1).detach()
        loss_warp_sds_1 = 0.5 * torch.nn.functional.mse_loss(rendered_image_latents.float(), target.float(), reduction='mean')
        
        warp_sds_grad_2 = torch.nan_to_num(warp_sds_grad_2)
        init_rendered_image_2 = self.pipe.image_processor.preprocess(rendered_image.to(self.weights_dtype), height=height, width=width, crops_coords=None, resize_mode='default')
        rendered_image_latents_2 = self.pipe._encode_vae_image(image=init_rendered_image_2.to(self.weights_dtype), generator=None)
        target_2 = (rendered_image_latents_2 - warp_sds_grad_2).detach()
        loss_warp_sds_2 = 0.5 * torch.nn.functional.mse_loss(rendered_image_latents_2.float(), target_2.float(), reduction='mean')
        
        return loss_warp_sds_1.to(torch.float32), loss_warp_sds_2.to(torch.float32)
    
    def cal_sds_grad(
        self,
        image: torch.FloatTensor, # rgb_BCHW: warped_image
        mask_image: torch.FloatTensor, # depth_B1HW: mask_image
        rendered_image: torch.FloatTensor, # rgb_BCHW: rendered_image
        num_inference_steps: int,
        guidance_scale: float, 
        guidance_mask: torch.Tensor = None, 
        num_images_per_prompt: int = 1
    ):
            
        height, width = 512, 512
        
        mask_condition = self.pipe.mask_processor.preprocess(mask_image.to(self.weights_dtype), height=height, width=width, resize_mode='default', crops_coords=None)
        init_warped_image = self.pipe.image_processor.preprocess(image.to(self.weights_dtype), height=height, width=width, crops_coords=None, resize_mode='default')
        masked_image_latents = init_warped_image * (mask_condition < 0.5)
        
        with torch.no_grad():
            warp_sds_grad = self.single_forward_grad_3(
                self.pipe, 
                self.pipe_sd, 
                t_start=0.25, 
                t_end=0.75, 
                prompt=None, 
                prompt_embeds=self.prompt_embeds, 
                negative_prompt_embeds=None,
                image=rendered_image.to(self.weights_dtype),
                mask_image=mask_condition.to(self.weights_dtype),  
                guidance_scale=guidance_scale,   
                height=height, 
                width=width, 
                masked_image_latents=masked_image_latents.to(self.weights_dtype), 
                cross_attention_kwargs={"scale": 1.0}
            )
        
        warp_sds_grad = torch.nan_to_num(warp_sds_grad)
        init_rendered_image = self.pipe.image_processor.preprocess(rendered_image.to(self.weights_dtype), height=height, width=width, crops_coords=None, resize_mode='default')
        rendered_image_latents = self.pipe._encode_vae_image(image=init_rendered_image.to(self.weights_dtype), generator=None)
        target = (rendered_image_latents - warp_sds_grad).detach()
        if guidance_mask:
            guidance_mask_condition = self.pipe.mask_processor.preprocess(guidance_mask.to(self.weights_dtype), height=height, width=width, resize_mode='default', crops_coords=None)
            guidance_mask_condition = torch.nn.functional.interpolate(
                guidance_mask_condition, size=(height // self.pipe.vae_scale_factor, width // self.pipe.vae_scale_factor)
            )
            rendered_image_latents[guidance_mask_condition] = 0.
            target[guidance_mask_condition] = 0.
        loss_warp_sds = 0.5 * torch.nn.functional.mse_loss(rendered_image_latents.float(), target.float(), reduction='mean')
        
        return loss_warp_sds.to(torch.float32)

    def cal_sds_ori_grad(
        self,
        image: torch.FloatTensor, # rgb_BCHW: warped_image
        mask_image: torch.FloatTensor, # depth_B1HW: mask_image
        rendered_image: torch.FloatTensor, # rgb_BCHW: rendered_image
        num_inference_steps: int,
        guidance_scale: float, 
        guidance_mask: torch.Tensor = None, 
        num_images_per_prompt: int = 1
    ):
            
        height, width = 512, 512
        
        with torch.no_grad():
            warp_sds_grad = self.single_forward_grad_4(
                self.pipe_sd, 
                t_start=0.25, 
                t_end=0.75, 
                prompt=None, 
                prompt_embeds=self.prompt_embeds, 
                negative_prompt_embeds=None,
                image=rendered_image.to(self.weights_dtype),
                guidance_scale=guidance_scale,   
                height=height, 
                width=width, 
                cross_attention_kwargs={"scale": 1.0}
            )
        
        warp_sds_grad = torch.nan_to_num(warp_sds_grad)
        init_rendered_image = self.pipe_sd.image_processor.preprocess(rendered_image.to(self.weights_dtype), height=height, width=width, crops_coords=None, resize_mode='default')
        rendered_image_latents = self.pipe_sd.vae.encode(init_rendered_image.to(self.weights_dtype)).latent_dist.sample() * self.pipe_sd.vae.config.scaling_factor
        target = (rendered_image_latents - warp_sds_grad).detach()
        loss_warp_sds = 0.5 * torch.nn.functional.mse_loss(rendered_image_latents.float(), target.float(), reduction='mean')
        
        return loss_warp_sds.to(torch.float32)
    
    def single_forward_grad(
        self,
        pipe, 
        t_start = 0.0, 
        t_end = 1.0, 
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.FloatTensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        **kwargs,
    ):
        height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor
        self.cfg.guidance_scale = guidance_scale
        pipe._guidance_scale = guidance_scale
        pipe._clip_skip = clip_skip
        pipe._cross_attention_kwargs = cross_attention_kwargs
        pipe._interrupt = False
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = pipe._execution_device
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=pipe.clip_skip,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        if padding_mask_crop is not None:
            crops_coords = pipe.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"
        init_image = pipe.image_processor.preprocess(
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        num_channels_latents = pipe.vae.config.latent_channels
        num_channels_unet = pipe.unet.config.in_channels
        latents = pipe._encode_vae_image(image=init_image, generator=generator)
        latents = latents.repeat(batch_size // latents.shape[0], 1, 1, 1)
        mask_condition = pipe.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents
        mask, masked_image_latents = pipe.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            self.do_classifier_free_guidance,
        )
        if num_channels_unet == 9:
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != pipe.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {pipe.unet.config} expects"
                    f" {pipe.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {pipe.unet.__class__} should have either 4 or 9 input channels, not {pipe.unet.config.in_channels}."
            )
        added_cond_kwargs = None
        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        B = latents.shape[0]
        latents = latents.detach()
        t = torch.randint(
            int(self.num_train_timesteps * t_start),
            int(self.num_train_timesteps * t_end),
            [B],
            dtype=torch.long,
            device=self.device,
        )
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        latent_model_input = torch.cat([noisy_latents] * 2) if self.do_classifier_free_guidance else noisy_latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        if num_channels_unet == 9:
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)
        warp_sds_grad = self.alphas[t] * (noise_pred - noise) 
        return warp_sds_grad
    
    def single_forward_grad_2(
        self,
        pipe, 
        pipe_sd, 
        t_start = 0.0, 
        t_end = 1.0, 
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.FloatTensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        **kwargs,
    ):
        height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

        self.cfg.guidance_scale = guidance_scale # NOTICE
        pipe._guidance_scale = guidance_scale
        pipe._clip_skip = clip_skip
        pipe._cross_attention_kwargs = cross_attention_kwargs
        pipe._interrupt = False
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = pipe._execution_device
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=pipe.clip_skip,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        if padding_mask_crop is not None:
            crops_coords = pipe.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"
        init_image = pipe.image_processor.preprocess(
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        num_channels_latents = pipe.vae.config.latent_channels
        num_channels_unet = pipe.unet.config.in_channels
        latents = pipe._encode_vae_image(image=init_image, generator=generator)
        latents = latents.repeat(batch_size // latents.shape[0], 1, 1, 1)
        mask_condition = pipe.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )

        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents

        mask, masked_image_latents = pipe.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            self.do_classifier_free_guidance,
        )
        if num_channels_unet == 9:
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != pipe.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {pipe.unet.config} expects"
                    f" {pipe.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {pipe.unet.__class__} should have either 4 or 9 input channels, not {pipe.unet.config.in_channels}."
            )

        added_cond_kwargs = None
        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        
        B = latents.shape[0]
        latents = latents.detach()
        t = torch.randint(
            int(self.num_train_timesteps * t_start),
            int(self.num_train_timesteps * t_end),
            [B],
            dtype=torch.long,
            device=self.device,
        )
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        latent_model_input = torch.cat([noisy_latents] * 2) if self.do_classifier_free_guidance else noisy_latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        if num_channels_unet == 9:
            latent_model_input_1 = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            noise_pred = pipe.unet(
                latent_model_input_1,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        if self.use_sd15:
            latent_model_input_2 = latent_model_input
            noise_pred_sd = pipe_sd.unet(
                latent_model_input_2,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            if self.do_classifier_free_guidance:
                noise_pred_sd_uncond, noise_pred_sd_text = noise_pred_sd.chunk(2)
                noise_pred_sd = noise_pred_sd_uncond + pipe.guidance_scale * (noise_pred_sd_text - noise_pred_sd_uncond)
        
        warp_sds_grad_1 = self.alphas[t] * (noise_pred - noise)
        warp_sds_grad_2 = self.alphas[t] * (noise_pred - noise_pred_sd)
        return warp_sds_grad_1, warp_sds_grad_2
    
    def single_forward_grad_3(
        self,
        pipe, 
        pipe_sd, 
        t_start = 0.0, 
        t_end = 1.0, 
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.FloatTensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        **kwargs,
    ):
        height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor
        self.cfg.guidance_scale = guidance_scale
        pipe._guidance_scale = guidance_scale
        pipe._clip_skip = clip_skip
        pipe._cross_attention_kwargs = cross_attention_kwargs
        pipe._interrupt = False
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = pipe._execution_device
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=pipe.clip_skip,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        if padding_mask_crop is not None:
            crops_coords = pipe.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"
        init_image = pipe.image_processor.preprocess(
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        num_channels_latents = pipe.vae.config.latent_channels
        num_channels_unet = pipe.unet.config.in_channels
        latents = pipe._encode_vae_image(image=init_image, generator=generator)
        latents = latents.repeat(batch_size // latents.shape[0], 1, 1, 1)
        mask_condition = pipe.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )

        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents

        mask, masked_image_latents = pipe.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            self.do_classifier_free_guidance,
        )
        if num_channels_unet == 9:
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != pipe.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {pipe.unet.config} expects"
                    f" {pipe.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {pipe.unet.__class__} should have either 4 or 9 input channels, not {pipe.unet.config.in_channels}."
            )

        added_cond_kwargs = None
        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        B = latents.shape[0]
        latents = latents.detach()
        t = torch.randint(
            int(self.num_train_timesteps * t_start),
            int(self.num_train_timesteps * t_end),
            [B],
            dtype=torch.long,
            device=self.device,
        )
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        latent_model_input = torch.cat([noisy_latents] * 2) if self.do_classifier_free_guidance else noisy_latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                
        if self.use_sd15:
            latent_model_input_2 = latent_model_input
            noise_pred_sd = pipe_sd.unet(
                latent_model_input_2,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            if self.do_classifier_free_guidance:
                noise_pred_sd_uncond, noise_pred_sd_text = noise_pred_sd.chunk(2)
                noise_pred_sd = noise_pred_sd_uncond + pipe.guidance_scale * (noise_pred_sd_text - noise_pred_sd_uncond)
        
        warp_sds_grad = self.alphas[t] * (noise_pred_sd - noise) 
        return warp_sds_grad
    
    def single_forward_grad_4(
        self,
        pipe_sd, 
        t_start = 0.0, 
        t_end = 1.0, 
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        **kwargs,
    ):

        height = height or pipe_sd.unet.config.sample_size * pipe_sd.vae_scale_factor
        width = width or pipe_sd.unet.config.sample_size * pipe_sd.vae_scale_factor

        self.cfg.guidance_scale = guidance_scale
        pipe_sd._guidance_scale = guidance_scale
        pipe_sd._clip_skip = clip_skip
        pipe_sd._cross_attention_kwargs = cross_attention_kwargs
        pipe_sd._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = pipe_sd._execution_device

        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=pipe_sd.clip_skip,
        )
        
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        crops_coords = None
        resize_mode = "default"
        init_image = pipe_sd.image_processor.preprocess(
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        ) 
        latents = pipe_sd.vae.encode(init_image).latent_dist.sample() * pipe_sd.vae.config.scaling_factor
        latents = latents.repeat(batch_size // latents.shape[0], 1, 1, 1)
        added_cond_kwargs = None
        timestep_cond = None
        if pipe_sd.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(pipe_sd.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = pipe_sd.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe_sd.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        B = latents.shape[0]
        latents = latents.detach()
        t = torch.randint(
            int(self.num_train_timesteps * t_start),
            int(self.num_train_timesteps * t_end),
            [B],
            dtype=torch.long,
            device=self.device,
        )
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_sd.add_noise(latents, noise, t)
        latent_model_input = torch.cat([noisy_latents] * 2) if self.do_classifier_free_guidance else noisy_latents
        latent_model_input = pipe_sd.scheduler.scale_model_input(latent_model_input, t)
                
        if self.use_sd15:
            latent_model_input_2 = latent_model_input
            noise_pred_sd = pipe_sd.unet(
                latent_model_input_2,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipe_sd.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            if self.do_classifier_free_guidance:
                noise_pred_sd_uncond, noise_pred_sd_text = noise_pred_sd.chunk(2)
                noise_pred_sd = noise_pred_sd_uncond + pipe_sd.guidance_scale * (noise_pred_sd_text - noise_pred_sd_uncond)
        
        warp_sds_grad = self.alphas[t] * (noise_pred_sd - noise) 
        return warp_sds_grad
