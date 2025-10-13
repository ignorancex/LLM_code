import logging
from typing import Dict, Optional, Union, List

from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as TF
import torchvision.transforms.functional as tf
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


class MatSwapPipeline(StableDiffusionPipeline):
    @classmethod
    def from_pretrained(cls, pretrained_model_path, *args, **kwargs):
        # Load the base pipeline
        pipe = super().from_pretrained(pretrained_model_path, *args, **kwargs)

        # Use placeholder weight file in "ip-adapter format"
        pipe.load_ip_adapter(
            pretrained_model_name_or_path_or_dict=pretrained_model_path,
            subfolder='ip_adapter',
            weight_name='placeholder.safetensors',
            image_encoder_folder=None,
        )

        # Load the actual MatSwap adapter weights
        from safetensors.torch import load_file
        ip_weight_path = Path(pretrained_model_path) / 'ip_adapter'/ 'matswap_adapter.safetensors'
        state = load_file(ip_weight_path)
        pipe.unet.load_state_dict(state, strict=False)
        return pipe

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler,
        safety_checker,
        feature_extractor,
        image_encoder,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae, text_encoder, tokenizer, unet, scheduler,
            safety_checker, feature_extractor, image_encoder,
            requires_safety_checker
        )
        # from pathlib import Path
        # self.ip_adapter_path = Path("IP-Adapter/models/ip-adapter_sd15.bin")
        # *path, subfolder, filename = self.ip_adapter_path.parts
        # assert self.ip_adapter_path.is_file(), \
        #     f"IP adapter weights {self.ip_adapter_path} does not exist."
        # assert self.ip_adapter_path.with_name('image_encoder').is_dir(), \
        #     f"IP adapter image encoder folder does not exist."

        # self.load_ip_adapter(
        #     pretrained_model_name_or_path_or_dict=path,
        #     subfolder=subfolder,
        #     weight_name=filename,
        #     image_encoder_folder="image_encoder",
        # )
        # self.replace_unet_conv_in_()
        # self.replace_unet_for_inpainting_()

    def replace_unet_conv_in_(self) -> torch.Tensor:
        self.rgb_weight = self.unet.conv_in.weight # [320, 4, 3, 3]
        _bias = self.unet.conv_in.bias.clone()  # [320]
        _zero_weight = torch.zeros_like(self.rgb_weight)

        new_weight = []

        new_weight.append(_zero_weight.clone())                 # compressed normals is 4D
        new_weight.append(_zero_weight.clone()[:, :1])      # irradiance is 1D
        new_weight.append(self.rgb_weight.clone())                  # image latent is 4D

        _weight = torch.cat(new_weight, dim=1)

        # new conv_in channel
        _n_convin_out_channel = self.unet.conv_in.out_channels
        _new_conv_in = nn.Conv2d(
            9, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = nn.Parameter(_weight)
        _new_conv_in.bias = nn.Parameter(_bias)
        self.unet.conv_in = _new_conv_in

    def replace_unet_for_inpainting_(self) -> None:
        # _old_in_channels = self.in_channels
        # self.in_channels += 4 + (1*(3*self.latent_irr_mask+1))*(not self.mask_attn) # masked image latent and mask

        if hasattr(self, 'rgb_weight'):
            rgb_weight = self.rgb_weight.to(device=self.unet.conv_in.weight.device)
            del self.rgb_weight
        else:
            rgb_weight = self.unet.conv_in.weight

        conv_weights = [
            self.unet.conv_in.weight,               # current channel weights
            torch.zeros_like(rgb_weight),           # use zeros for RGB image
            # torch.zeros_like(rgb_weight[:,:1])      # use zeros for mask channel
        ]

        # if self.mask_attn:
        #     pass
        # else:
        #     if self.latent_irr_mask:
        #         conv_weights.append(torch.zeros_like(rgb_weight))
        #     else:
        conv_weights.append(torch.zeros_like(rgb_weight[:,:1]))

        new_weight = torch.cat(conv_weights, dim=1)

        ## Overwrite layer in unet and replace weights
        conv_in = nn.Conv2d(
            in_channels=14,
            out_channels=self.unet.conv_in.out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        conv_in.weight = nn.Parameter(new_weight)
        conv_in.bias = nn.Parameter(self.unet.conv_in.bias)
        self.unet.conv_in = conv_in

        # logging.info(f"Unet conv_in updated from {_old_in_channels} to {self.in_channels} for inpainting.")
        return None

    def encode_empty_text(self):
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        return self.text_encoder(text_input_ids)[0].to(self.dtype)

    @property
    def latent_size(self):
        return self._latent_size

    @torch.no_grad()
    def __call__(
        self,
        target_normals: torch.Tensor,
        target_irradiance: torch.Tensor,
        target_image: torch.Tensor,
        source_image: torch.Tensor,
        target_mask: torch.Tensor,
        denoising_steps: Optional[int] = 50,
        batch_size: int = 1,
        generator: Union[torch.Generator, None] = None,
        show_progress_bar: bool = True,
        guidance_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        device: Optional[torch.device] = None,
        output_type: str = "pil",
        **kwargs
    ):

        self._guidance_scale = guidance_scale
        self._device = device

        # 0. Denoising resolution
        resolution = (height, width) if height and width else None
        h, w = resolution if resolution else target_image.shape[-2:]
        self._latent_size = int(h//self.vae_scale_factor), int(w//self.vae_scale_factor)

        # 1. Encode input prompt
        real_batch_size = batch_size*(self.do_classifier_free_guidance+1)
        prompt_embeds = self.encode_empty_text().repeat(real_batch_size, 1, 1).to(device)

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(denoising_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare latent variables
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=4,
            latent_size=self.latent_size,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator
        )

        # 4. Prepare conditioning maps
        conditioning_stack, inpaint_stack, unet_kwargs = self.prepare_conditioning_maps(
            target_normals=target_normals,
            target_irradiance=target_irradiance,
            target_image=target_image,
            target_mask=target_mask,
            source_image=source_image,
            device=device,
        )

        if show_progress_bar:
            timesteps = tqdm(timesteps, total=len(timesteps), leave=False)

        # 5. Denoising loop
        for t in timesteps:
            # Concatenate conditioning stack with noisy latent
            latent_model_input = conditioning_stack + [latents] + inpaint_stack
            latent_model_input = torch.cat(latent_model_input, dim=1)

            # Expand latents for classifier-free guidance
            if self.do_classifier_free_guidance:
                latent_model_input = torch.cat([latent_model_input] * 2)

            # Concatenate conditioning maps with latent input
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise residual
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                **unet_kwargs,
            ).sample

            # Perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_exemplar = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_exemplar - noise_pred_uncond)

            # Compute previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 6. Decode latents to RGB
        img = self.decode_latents(latents)

        # 7. Post-process the decoded image
        out = self.image_processor.postprocess(img, output_type=output_type, do_denormalize=[True])

        return out

    def prepare_conditioning_maps(
        self,
        target_normals: torch.Tensor,
        target_irradiance: torch.Tensor,
        target_image: torch.Tensor,
        target_mask: torch.Tensor,
        source_image: torch.Tensor,
        device: torch.device,
    ):
        stack = []
        inpaint_stack = []

        # 1. Add geometry (normals)
        stack.append(
            self.encode_rgb(target_normals.to(device))
        )

        # 2. Add irradiance
        stack.append(
            tf.resize(
                img=target_irradiance.to(device),
                size=self.latent_size,
                interpolation=TF.InterpolationMode.BILINEAR,
            )
        )

        # 3. Encode target image
        target_image = target_image.to(device)
        target_image_latent = self.encode_rgb(target_image * 2 - 1)  # Scale to [-1, 1] and encode
        inpaint_stack.append(target_image_latent)

        # 4. Add mask for inpainting
        target_mask = target_mask if isinstance(target_mask, list) else [target_mask]
        target_mask_latent_or_down = [tf.resize(
            img=x.to(device),
            size=self.latent_size,
            interpolation=TF.InterpolationMode.NEAREST,
        ).float() for x in target_mask]

        assert len(target_mask_latent_or_down) == 1
        inpaint_stack.append(target_mask_latent_or_down[0])

        # 5. Add CLIP embedding for IP-Adapter
        source_image = source_image if isinstance(source_image, list) else [source_image]
        clip_images = self.feature_extractor.preprocess(
            images=torch.cat(source_image),
            do_rescale=False,
            return_tensors='pt'
        ).pixel_values.to(device)

        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image=[clip_images], # list size == nb of IP-Adapter
            ip_adapter_image_embeds=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )

        added_cond_kwargs={"image_embeds": image_embeds}

        unet_kwargs = dict(
            added_cond_kwargs=added_cond_kwargs,
        )
        return stack, inpaint_stack, unet_kwargs

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        h = self.vae.encoder(rgb_in) # encode
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.vae.config.scaling_factor # scale latent
        return rgb_latent

    def prepare_latents(self, batch_size, num_channels_latents, latent_size, dtype, device, generator):
        shape = (batch_size, num_channels_latents, *latent_size)
        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        # image = (image / 2 + 0.5).clamp(0, 1)
        return image