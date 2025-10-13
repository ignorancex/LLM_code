from typing import List
from tqdm import tqdm
import torch
from diffsynth.models import ModelManager
from diffsynth.controlnets import ControlNetConfigUnit
from diffsynth.prompters.flux_prompter import FluxPrompter
from diffsynth.pipelines.flux_image import FluxImagePipeline, lets_dance_flux, TeaCache



class FluxPrompterNexusGen(FluxPrompter):
    def encode_prompt(
        self,
        prompt,
        positive=True,
        device="cuda",
        t5_sequence_length=512,
        clip_only=False
    ):
        prompt = self.process_prompt(prompt, positive=positive)
        # CLIP
        pooled_prompt_emb = self.encode_prompt_using_clip(prompt, self.text_encoder_1, self.tokenizer_1, 77, device)
        if clip_only:
            return None, pooled_prompt_emb, None
        # T5
        prompt_emb = self.encode_prompt_using_t5(prompt, self.text_encoder_2, self.tokenizer_2, t5_sequence_length, device)
        # text_ids
        text_ids = torch.zeros(prompt_emb.shape[0], prompt_emb.shape[1], 3).to(device=device, dtype=prompt_emb.dtype)
        return prompt_emb, pooled_prompt_emb, text_ids


class NexusGenPipeline(FluxImagePipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompter = FluxPrompterNexusGen()

    def encode_prompt(self, prompt, positive=True, t5_sequence_length=512, clip_only=False):
        prompt_emb, pooled_prompt_emb, text_ids = self.prompter.encode_prompt(
            prompt, device=self.device, positive=positive, t5_sequence_length=t5_sequence_length, clip_only=clip_only
        )
        return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb, "text_ids": text_ids}


    def prepare_prompts(self, prompt, image_embed, t5_sequence_length, negative_prompt, cfg_scale):
        # Extend prompt
        self.load_models_to_device(['text_encoder_1', 'text_encoder_2'])

        # Encode prompts
        if image_embed is not None:
            image_embed = image_embed.to(self.torch_dtype)
            prompt_emb_posi = self.encode_prompt("", positive=True, clip_only=True)
            if len(image_embed.size()) == 2:
                image_embed = image_embed.unsqueeze(0)
            prompt_emb_posi['prompt_emb'] = image_embed
            prompt_emb_posi['text_ids'] = torch.zeros(image_embed.shape[0], image_embed.shape[1], 3).to(device=self.device, dtype=self.torch_dtype)
        else:
            prompt_emb_posi = self.encode_prompt(prompt, t5_sequence_length=t5_sequence_length)
        prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False, t5_sequence_length=t5_sequence_length) if cfg_scale != 1.0 else None
        return prompt_emb_posi, prompt_emb_nega

class NexusGenGenerationPipeline(NexusGenPipeline):

    @staticmethod
    def from_model_manager(model_manager: ModelManager, device=None, torch_dtype=None):
        pipe = NexusGenGenerationPipeline(
            device=model_manager.device if device is None else device,
            torch_dtype=model_manager.torch_dtype if torch_dtype is None else torch_dtype,
        )
        pipe.fetch_models(model_manager)
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt,
        negative_prompt="",
        cfg_scale=1.0,
        embedded_guidance=3.5,
        t5_sequence_length=512,
        # image_embed
        image_embed=None,
        # Image
        input_image=None,
        denoising_strength=1.0,
        height=1024,
        width=1024,
        seed=None,
        # Steps
        num_inference_steps=30,
        # TeaCache
        tea_cache_l1_thresh=None,
        # Tile
        tiled=False,
        tile_size=128,
        tile_stride=64,
        # Progress bar
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        height, width = self.check_resize_height_width(height, width)

        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        latents, _ = self.prepare_latents(input_image, height, width, seed, tiled, tile_size, tile_stride)

        # Prompt
        prompt_emb_posi, prompt_emb_nega = self.prepare_prompts(prompt, image_embed, t5_sequence_length, negative_prompt, cfg_scale)

        # Extra input
        extra_input = self.prepare_extra_input(latents, guidance=embedded_guidance)

        # TeaCache
        tea_cache_kwargs = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh) if tea_cache_l1_thresh is not None else None}

        # Denoise
        self.load_models_to_device(['dit'])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Positive side
            noise_pred_posi = lets_dance_flux(
                dit=self.dit, hidden_states=latents, timestep=timestep, 
                **prompt_emb_posi, **tiler_kwargs, **extra_input, **tea_cache_kwargs)
            # Classifier-free guidance
            if cfg_scale != 1.0:
                # Negative side
                noise_pred_nega = lets_dance_flux(
                    dit=self.dit, controlnet=self.controlnet,
                    hidden_states=latents, timestep=timestep,
                    **prompt_emb_nega, **tiler_kwargs, **extra_input,  **tea_cache_kwargs)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Iterate
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))
        
        # Decode image
        self.load_models_to_device(['vae_decoder'])
        image = self.decode_image(latents, **tiler_kwargs)

        # Offload all models
        self.load_models_to_device([])
        return image


class NexusGenEditingPipeline(NexusGenPipeline):
    @staticmethod
    def from_model_manager(model_manager: ModelManager, device=None, torch_dtype=None):
        pipe = NexusGenEditingPipeline(
            device=model_manager.device if device is None else device,
            torch_dtype=model_manager.torch_dtype if torch_dtype is None else torch_dtype,
        )
        pipe.fetch_models(model_manager)
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt,
        negative_prompt="",
        cfg_scale=1.0,
        embedded_guidance=3.5,
        t5_sequence_length=512,
        # image_embed
        image_embed=None,
        target_embed_size=18,
        ref_embed_size=36,
        # Image
        input_image=None,
        denoising_strength=1.0,
        height=1024,
        width=1024,
        seed=None,
        # Steps
        num_inference_steps=30,
        # TeaCache
        tea_cache_l1_thresh=None,
        # Tile
        tiled=False,
        tile_size=128,
        tile_stride=64,
        # Progress bar
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        height, width = self.check_resize_height_width(height, width)

        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        latents, _ = self.prepare_latents(input_image, height, width, seed, tiled, tile_size, tile_stride)

        # Prompt
        prompt_emb_posi, prompt_emb_nega = self.prepare_prompts(prompt, image_embed, t5_sequence_length, negative_prompt, cfg_scale)

        # prepare text ids for target and reference embeddings
        batch_size, height, width = latents.shape[0], target_embed_size, target_embed_size
        embed_ids = torch.zeros(height // 2, width // 2, 3)
        scale_factor_height, scale_factor_width = latents.shape[-2] / height, latents.shape[-1] / width
        embed_ids[..., 1] = embed_ids[..., 1] + torch.arange(height // 2)[:, None] * scale_factor_height
        embed_ids[..., 2] = embed_ids[..., 2] + torch.arange(width // 2)[None, :] * scale_factor_width
        embed_ids = embed_ids[None, :].repeat(batch_size, 1, 1, 1).reshape(batch_size, height // 2 * width // 2, 3)
        embed_text_ids = embed_ids.to(device=latents.device, dtype=latents.dtype)
        
        num_target_embeds = target_embed_size * target_embed_size // 4
        if prompt_emb_posi['prompt_emb'].shape[1] == num_target_embeds:
            prompt_emb_posi['text_ids'] = embed_text_ids
        else:
            batch_size, height, width = latents.shape[0], ref_embed_size, ref_embed_size
            ref_embed_ids = torch.zeros(height // 2, width // 2, 3)
            scale_factor_height, scale_factor_width = latents.shape[-2] / height, latents.shape[-1] / width
            ref_embed_ids[..., 0] = ref_embed_ids[..., 0] + 1.0
            ref_embed_ids[..., 1] = ref_embed_ids[..., 1] + torch.arange(height // 2)[:, None] * scale_factor_height
            ref_embed_ids[..., 2] = ref_embed_ids[..., 2] + torch.arange(width // 2)[None, :] * scale_factor_width
            ref_embed_ids = ref_embed_ids[None, :].repeat(batch_size, 1, 1, 1).reshape(batch_size, height // 2 * width // 2, 3)
            ref_embed_text_ids = ref_embed_ids.to(device=latents.device, dtype=latents.dtype)
            prompt_emb_posi['text_ids'] = torch.cat([embed_text_ids, ref_embed_text_ids], dim=1)

        # Extra input
        extra_input = self.prepare_extra_input(latents, guidance=embedded_guidance)

        # TeaCache
        tea_cache_kwargs = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh) if tea_cache_l1_thresh is not None else None}

        # Denoise
        self.load_models_to_device(['dit'])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Positive side
            noise_pred_posi = lets_dance_flux(
                dit=self.dit, hidden_states=latents, timestep=timestep, 
                **prompt_emb_posi, **tiler_kwargs, **extra_input, **tea_cache_kwargs)
            # Classifier-free guidance
            if cfg_scale != 1.0:
                # Negative side
                noise_pred_nega = lets_dance_flux(
                    dit=self.dit, controlnet=self.controlnet,
                    hidden_states=latents, timestep=timestep,
                    **prompt_emb_nega, **tiler_kwargs, **extra_input,  **tea_cache_kwargs)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Iterate
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))
        
        # Decode image
        self.load_models_to_device(['vae_decoder'])
        image = self.decode_image(latents, **tiler_kwargs)

        # Offload all models
        self.load_models_to_device([])
        return image
