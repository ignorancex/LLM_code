import typing as tp

import torch
from diffusers import AutoencoderKL, DDPMScheduler, Transformer2DModel
from torchvision import transforms
from transformers import T5EncoderModel, T5Tokenizer

from src.constants.dataset import DatasetColumns
from src.models.base_model import BaseModel
from src.models.stable_diffusion import StableDiffusion


class PixArt(StableDiffusion):
    def __init__(
        self,
        pretrained_model_name: str,
        revision: str | None = None,
    ) -> None:
        BaseModel.__init__(
            self, pretrained_model_name=pretrained_model_name, revision=revision
        )

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.timesteps = self.noise_scheduler.timesteps

        self.text_encoder = T5EncoderModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.revision,
        )

        self.tokenizer = T5Tokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.revision,
        )

        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="vae", revision=self.revision
        )

        self.unet = Transformer2DModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=self.revision,
        )

        self.image_processor = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.guidance_scale = 4.5

    def tokenize(self, caption: str) -> dict[str, tp.Any]:
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            DatasetColumns.tokenized_text.name: inputs.input_ids.squeeze(),
            DatasetColumns.text_attention_mask.name: inputs.attention_mask.squeeze(),
        }

    def _get_unconditional_text_input(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uncond_input = self.tokenizer(
            [""] * batch_size,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return uncond_input.input_ids.squeeze(), uncond_input.attention_mask.squeeze()

    def get_encoder_hidden_states(
        self, batch: dict[str, torch.Tensor], do_classifier_free_guidance: bool = False
    ) -> torch.Tensor:
        text_input_ids = batch[DatasetColumns.tokenized_text.name]
        text_attention_mask = batch[DatasetColumns.text_attention_mask.name]

        if do_classifier_free_guidance:
            (
                uncond_input_ids,
                uncond_attention_mask,
            ) = self._get_unconditional_text_input(text_input_ids.shape[0])
            text_input_ids = torch.cat(
                [uncond_input_ids.to(text_input_ids.device), text_input_ids]
            )
            text_attention_mask = torch.cat(
                [uncond_input_ids.to(text_input_ids.device), text_attention_mask]
            )
        batch[DatasetColumns.text_attention_mask.name + "_"] = text_attention_mask
        return self.text_encoder(text_input_ids, text_attention_mask)[0].to(
            torch.float16
        )

    def predict_next_latents(
        self,
        latents: torch.Tensor,
        timestep_index: int,
        encoder_hidden_states: torch.Tensor,
        batch: dict[str, torch.Tensor],
        return_pred_original: bool = False,
        do_classifier_free_guidance: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = self.timesteps[timestep_index]
        latent_model_input = self.noise_scheduler.scale_model_input(
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents,
            timestep,
        )
        noise_pred = self.unet(
            latent_model_input,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=batch[DatasetColumns.text_attention_mask.name + "_"],
            timestep=timestep.expand(latent_model_input.shape[0]),
            added_cond_kwargs={"resolution": None, "aspect_ratio": None},
        ).sample.chunk(2, dim=1)[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if return_pred_original:
            pred_original_sample = self.noise_scheduler.step(
                noise_pred, timestep, latents
            ).pred_original_sample

            return pred_original_sample, noise_pred

        latents = self.noise_scheduler.step(noise_pred, timestep, latents).prev_sample

        return latents, noise_pred
