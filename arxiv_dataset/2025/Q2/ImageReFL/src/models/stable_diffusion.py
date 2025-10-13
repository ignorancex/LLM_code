import math
import random
import typing as tp

import torch
from diffusers import AutoencoderKL, DDPMScheduler, SchedulerMixin, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import EMAModel
from peft import LoraConfig
from PIL import Image
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

from src.constants.dataset import DatasetColumns
from src.models.base_model import BaseModel


class StableDiffusion(BaseModel):
    """
    A Stable Diffusion model wrapper that provides functionality for text-to-image synthesis,
    noise scheduling, latent space manipulation, and image decoding.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        revision: str | None = None,
        noise_scheduler: SchedulerMixin | None = None,
        guidance_scale: float = 7.5,
        use_ema: bool = False,
        use_lora: bool = False,
        lora_rank: int | None = None,
        use_image_shifting: bool = False,
    ) -> None:
        """
        Initializes the components of  StableDiffusion model.

        Args:
            pretrained_model_name (str): The name or path of the pretrained model to use.
            revision (str | None): The specific revision of the model to use. Defaults to None.

        Notes:
            The Variational Autoencoder (VAE) and text encoder parameters are frozen to prevent gradient updates.
        """
        super().__init__(pretrained_model_name=pretrained_model_name, revision=revision)

        self.noise_scheduler = (
            noise_scheduler
            if noise_scheduler is not None
            else DDPMScheduler.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="scheduler",
            )
        )
        self.timesteps = self.noise_scheduler.timesteps

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.revision,
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.revision,
        )

        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.revision,
        )

        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="unet",
        )

        self.use_ema = use_ema
        if use_ema:
            ema_unet = UNet2DConditionModel.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="unet"
            )
            self.ema_unet = EMAModel(
                ema_unet.parameters(),
                model_cls=UNet2DConditionModel,
                model_config=ema_unet.config,
            )

        self.use_lora = use_lora
        if use_lora:
            self.lora_rank = lora_rank
            self.unet.requires_grad_(False)
            unet_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.unet.add_adapter(unet_lora_config)

        self.reward_image_processor = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        # reproduce StableDiffusionPipeline
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.inference_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor
        )

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.guidance_scale = guidance_scale
        self.resolution = 512
        self.use_image_shifting = use_image_shifting

    @staticmethod
    def _shift_tensor_batch(images, dx=0, dy=0, fill_value=0):
        shifted = torch.roll(images, shifts=(dy, dx), dims=(2, 3))  # dims: H, W

        if dy > 0:
            shifted[:, :, :dy, :] = fill_value
        elif dy < 0:
            shifted[:, :, dy:, :] = fill_value

        if dx > 0:
            shifted[:, :, :, :dx] = fill_value
        elif dx < 0:
            shifted[:, :, :, dx:] = fill_value

        return shifted

    def train(self, mode: bool = True):
        """
        Sets the U-Net model to training mode.

        Args:
            mode (bool): Whether to enable or disable training mode.
        """
        self.unet.train()

    def eval(self, mode: bool = True):
        """
        Sets the U-Net model to evaluation mode.

        Args:
            mode (bool): Whether to enable or disable evaluation mode.
        """
        self.unet.eval()

    def tokenize(self, caption: str) -> dict[str, tp.Any]:
        """
        Tokenizes the given caption.

        Args:
            caption (str): The text to tokenize.

        Returns:
            dict: A dictionary containing the tokenized text tensor.
        """
        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            DatasetColumns.tokenized_text.name: input_ids,
        }

    def set_timesteps(self, max_timestep: int, device) -> None:
        """
        Sets the timesteps for the noise scheduler.

        Args:
            max_timestep (int): The maximum timestep for the diffusion process.
            device (torch.device): The device to set the timesteps on.
        """
        self.noise_scheduler.set_timesteps(max_timestep, device=device)
        self.timesteps = self.noise_scheduler.timesteps

    def get_noisy_latents_from_images(
        self,
        images: torch.Tensor,
        timestep_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Adds noise to the latents obtained from the input images.

        Args:
            images (torch.Tensor): Input image tensor.
            timestep_index (int): Index of the timestep for adding noise.

        Returns:
            tuple: A tuple containing the noisy latents and the noise tensor.
        """
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        noisy_images = self.noise_scheduler.add_noise(
            original_samples=latents,
            noise=noise,
            timesteps=(
                torch.ones((images.shape[0],), device=latents.device).int()
                * self.timesteps[timestep_index]
            ),
        )
        return noisy_images, noise

    def _get_negative_prompts(self, batch_size: int) -> torch.Tensor:
        return self.tokenizer(
            [""] * batch_size,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

    def get_encoder_hidden_states(
        self, batch: dict[str, torch.Tensor], do_classifier_free_guidance: bool = False
    ) -> torch.Tensor:
        """
        Retrieves the hidden states from the text encoder.

        Args:
            batch (dict): A batch containing tokenized text.
            do_classifier_free_guidance (bool)  Whether to do classifier free guidance

        Returns:
            torch.Tensor: Hidden states from the text encoder.
        """
        text_input = batch[DatasetColumns.tokenized_text.name]

        if do_classifier_free_guidance:
            text_input = torch.cat(
                [
                    self._get_negative_prompts(text_input.shape[0]).to(
                        text_input.device
                    ),
                    text_input,
                ]
            )

        return self.text_encoder(text_input)[0]

    def _get_unet_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: int,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return unet noise prediction

        Args:
            latent_model_input (torch.Tensor): Unet latents input
            timestep (int): noise scheduler timestep
            encoder_hidden_states (torch.Tensor): Text encoder hidden states

        Returns:
            torch.Tensor: noise prediction
        """

        return self.unet(
            latent_model_input,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

    def get_noise_prediction(
        self,
        latents: torch.Tensor,
        timestep_index: int,
        encoder_hidden_states: torch.Tensor,
        do_classifier_free_guidance: bool = False,
        detach_main_path: bool = False,
    ):
        """
        Return noise prediction

        Args:
            latents (torch.Tensor): Image latents
            timestep_index (int): noise scheduler timestep index
            encoder_hidden_states (torch.Tensor): Text encoder hidden states
            do_classifier_free_guidance (bool)  Whether to do classifier free guidance
            detach_main_path (bool): Detach gradient

        Returns:
            torch.Tensor: noise prediction
        """
        timestep = self.timesteps[timestep_index]

        latent_model_input = self.noise_scheduler.scale_model_input(
            sample=torch.cat([latents] * 2) if do_classifier_free_guidance else latents,
            timestep=timestep,
        )

        noise_pred = self._get_unet_prediction(
            latent_model_input=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            if detach_main_path:
                noise_pred_text = noise_pred_text.detach()

            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        return noise_pred

    def sample_next_latents(
        self,
        latents: torch.Tensor,
        timestep_index: int,
        noise_pred: torch.Tensor,
        return_pred_original: bool = False,
    ) -> torch.Tensor:
        """
        Return next latents prediction

        Args:
            latents (torch.Tensor): Image latents
            timestep_index (int): noise scheduler timestep index
            noise_pred (torch.Tensor): noise prediction
            return_pred_original (bool)  Whether to sample original sample

        Returns:
            torch.Tensor: latent prediction
        """
        timestep = self.timesteps[timestep_index]
        sample = self.noise_scheduler.step(
            model_output=noise_pred, timestep=timestep, sample=latents
        )
        return (
            sample.pred_original_sample if return_pred_original else sample.prev_sample
        )

    def predict_next_latents(
        self,
        latents: torch.Tensor,
        timestep_index: int,
        encoder_hidden_states: torch.Tensor,
        batch: dict[str, torch.Tensor],
        return_pred_original: bool = False,
        do_classifier_free_guidance: bool = False,
        detach_main_path: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the next latent states during the diffusion process.

        Args:
            latents (torch.Tensor): Current latent states.
            timestep_index (int): Index of the current timestep.
            encoder_hidden_states (torch.Tensor): Encoder hidden states from the text encoder.
            batch (dict): Input batch containing data.
            return_pred_original (bool): Whether to return the predicted original sample.
            do_classifier_free_guidance (bool)  Whether to do classifier free guidance
            detach_main_path (bool): Detach gradient

        Returns:
            tuple: Next latents and predicted noise tensor.
        """

        noise_pred = self.get_noise_prediction(
            latents=latents,
            timestep_index=timestep_index,
            encoder_hidden_states=encoder_hidden_states,
            do_classifier_free_guidance=do_classifier_free_guidance,
            detach_main_path=detach_main_path,
        )

        latents = self.sample_next_latents(
            latents=latents,
            noise_pred=noise_pred,
            timestep_index=timestep_index,
            return_pred_original=return_pred_original,
        )

        return latents, noise_pred

    def get_latents(self, batch_size: int, device: torch.device) -> torch.Tensor:
        latent_resolution = int(self.resolution) // self.vae_scale_factor
        return torch.randn(
            (
                batch_size,
                self.unet.config.in_channels,
                latent_resolution,
                latent_resolution,
            ),
            device=device,
        )

    def do_k_diffusion_steps(
        self,
        start_timestep_index: int,
        end_timestep_index: int,
        batch: dict[str, torch.Tensor],
        latents: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        return_pred_original: bool = False,
        do_classifier_free_guidance: bool = False,
        detach_main_path: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs multiple diffusion steps between specified timesteps.

        Args:
            start_timestep_index (int): Starting timestep index.
            end_timestep_index (int): Ending timestep index.
            batch (dict): Input batch containing data.
            latents (torch.Tensor | None): Initial latents (optional).
            encoder_hidden_states (torch.Tensor | None): Encoder hidden states (optional).
            return_pred_original (bool): Whether to return the predicted original sample.
            do_classifier_free_guidance (bool)  Whether to do classifier free guidance
            detach_main_path (bool): Detach gradient

        Returns:
            tuple: Resulting latents and encoder hidden states.
        """
        assert start_timestep_index <= end_timestep_index
        batch_size = batch[DatasetColumns.tokenized_text.name].size(0)
        device = batch[DatasetColumns.tokenized_text.name].device

        if encoder_hidden_states is None:
            encoder_hidden_states = self.get_encoder_hidden_states(
                batch=batch, do_classifier_free_guidance=do_classifier_free_guidance
            )

        if latents is None:
            latents = self.get_latents(batch_size=batch_size, device=device)

        for timestep_index in range(start_timestep_index, end_timestep_index - 1):
            latents, _ = self.predict_next_latents(
                latents=latents,
                timestep_index=timestep_index,
                encoder_hidden_states=encoder_hidden_states,
                batch=batch,
                return_pred_original=False,
                do_classifier_free_guidance=do_classifier_free_guidance,
                detach_main_path=detach_main_path,
            )
        res, _ = self.predict_next_latents(
            latents=latents,
            timestep_index=end_timestep_index - 1,
            encoder_hidden_states=encoder_hidden_states,
            batch=batch,
            return_pred_original=return_pred_original,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        return res, encoder_hidden_states

    def get_pil_image(self, raw_images: torch.Tensor) -> list[Image]:
        do_denormalize = [True] * raw_images.shape[0]
        images = self.inference_image_processor.postprocess(
            raw_images, output_type="pil", do_denormalize=do_denormalize
        )
        return images

    def get_reward_image(self, raw_images: torch.Tensor) -> torch.Tensor:
        reward_images = (raw_images / 2 + 0.5).clamp(0, 1)

        if self.use_image_shifting:
            self._shift_tensor_batch(
                reward_images,
                dx=random.randint(0, math.ceil(self.resolution / 224)),
                dy=random.randint(0, math.ceil(self.resolution / 224)),
            )

        return self.reward_image_processor(reward_images)

    def sample_image(
        self,
        latents: torch.Tensor | None,
        start_timestep_index: int,
        end_timestep_index: int,
        batch: dict[str, torch.Tensor],
        encoder_hidden_states: torch.Tensor | None = None,
        do_classifier_free_guidance: bool = False,
        detach_main_path: bool = False,
    ) -> torch.Tensor:
        """
        Generates an image sample by decoding the latents.

        Args:
            latents (torch.Tensor | None): Initial latents (optional).
            start_timestep_index (int): Starting timestep index.
            end_timestep_index (int): Ending timestep index.
            batch (dict): Input batch containing data.
            encoder_hidden_states (torch.Tensor | None): Encoder hidden states (optional).
            do_classifier_free_guidance (bool)  Whether to do classifier free guidance
            detach_main_path (bool): Detach gradient

        Returns:
            torch.Tensor: Decoded image sample.
        """
        pred_original_sample, _ = self.do_k_diffusion_steps(
            latents=latents,
            start_timestep_index=start_timestep_index,
            end_timestep_index=end_timestep_index,
            batch=batch,
            encoder_hidden_states=encoder_hidden_states,
            return_pred_original=True,
            do_classifier_free_guidance=do_classifier_free_guidance,
            detach_main_path=detach_main_path,
        )

        pred_original_sample /= self.vae.config.scaling_factor

        raw_image = self.vae.decode(pred_original_sample).sample
        return self.get_reward_image(raw_image)

    def sample_image_inference(
        self,
        latents: torch.Tensor | None,
        start_timestep_index: int,
        end_timestep_index: int,
        batch: dict[str, torch.Tensor],
        encoder_hidden_states: torch.Tensor | None = None,
        do_classifier_free_guidance: bool = False,
    ) -> tuple[torch.Tensor, list[Image]]:
        """
        Generates an image sample by decoding the latents.
        Returns image with original shape and resized to reward model

        Args:
            latents (torch.Tensor | None): Initial latents (optional).
            start_timestep_index (int): Starting timestep index.
            end_timestep_index (int): Ending timestep index.
            batch (dict): Input batch containing data.
            encoder_hidden_states (torch.Tensor | None): Encoder hidden states (optional).
            do_classifier_free_guidance (bool)  Whether to do classifier free guidance

        Returns:
            torch.Tensor: Decoded image sample.
        """
        pred_original_sample, _ = self.do_k_diffusion_steps(
            latents=latents,
            start_timestep_index=start_timestep_index,
            end_timestep_index=end_timestep_index,
            batch=batch,
            encoder_hidden_states=encoder_hidden_states,
            return_pred_original=True,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        pred_original_sample /= self.vae.config.scaling_factor

        raw_image = self.vae.decode(pred_original_sample).sample
        return self.get_reward_image(raw_image), self.get_pil_image(raw_image)
