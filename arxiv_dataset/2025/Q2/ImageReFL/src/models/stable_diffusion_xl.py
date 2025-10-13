import typing as tp

import torch
from diffusers import SchedulerMixin
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from src.constants.dataset import DatasetColumns
from src.models.stable_diffusion import StableDiffusion

HIDDEN_STATE_TYPE = tuple[torch.Tensor, torch.Tensor]


class StableDiffusionXL(StableDiffusion):
    """
    A Stable Diffusion XL model wrapper that provides functionality for text-to-image synthesis,
    noise scheduling, latent space manipulation, and image decoding.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        revision: str | None = None,
        noise_scheduler: SchedulerMixin | None = None,
        guidance_scale: float = 5,
        resolution: int = 512,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained_model_name=pretrained_model_name,
            revision=revision,
            noise_scheduler=noise_scheduler,
            guidance_scale=guidance_scale,
            **kwargs,
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name, subfolder="text_encoder_2", revision=revision
        )
        self.text_encoder_2.requires_grad_(False)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            pretrained_model_name, subfolder="tokenizer_2", revision=revision
        )
        self.second_tokenizer_suffix = "_2"
        self.resolution = resolution

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

        input_ids_2 = self.tokenizer_2(
            caption,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            DatasetColumns.tokenized_text.name: input_ids,
            DatasetColumns.tokenized_text.name
            + self.second_tokenizer_suffix: input_ids_2,
        }

    def _get_negative_prompts(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_ids_1 = self.tokenizer(
            [""] * batch_size,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        input_ids_2 = self.tokenizer_2(
            [""] * batch_size,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return inputs_ids_1, input_ids_2

    def get_encoder_hidden_states(
        self, batch: dict[str, torch.Tensor], do_classifier_free_guidance: bool = False
    ) -> HIDDEN_STATE_TYPE:
        """
        Retrieves the hidden states from the text encoder.

        Args:
            batch (dict): A batch containing tokenized text.
            do_classifier_free_guidance (bool)  Whether to do classifier free guidance

        Returns:
            torch.Tensor: Hidden states from the text encoder.
        """
        text_input_ids_list = [
            batch[DatasetColumns.tokenized_text.name],
            batch[DatasetColumns.tokenized_text.name + self.second_tokenizer_suffix],
        ]
        batch_size = text_input_ids_list[0].size(0)

        if do_classifier_free_guidance:
            negative_prompts = [
                embed.to(text_input_ids_list[0].device)
                for embed in self._get_negative_prompts(batch_size)
            ]

            text_input_ids_list = [
                torch.cat(
                    [
                        negative_prompt,
                        text_input,
                    ]
                )
                for text_input, negative_prompt in zip(
                    text_input_ids_list, negative_prompts
                )
            ]
        prompt_embeds_list = []

        text_encoders = [self.text_encoder, self.text_encoder_2]
        for text_encoder, text_input_ids in zip(text_encoders, text_input_ids_list):
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
                return_dict=False,
            )

            # Note: We are only ALWAYS interested in the pooled output of the final text encoder
            # (batch_size, pooled_dim)
            pooled_prompt_embeds = prompt_embeds[0]
            # (batch_size, seq_len, dim)
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        # (batch_size, seq_len, dim)
        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        # (batch_size, pooled_dim)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    def _get_unet_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: int,
        encoder_hidden_states: HIDDEN_STATE_TYPE,
    ) -> torch.Tensor:
        """
        Return unet noise prediction

        Args:
            latent_model_input (torch.Tensor): Unet latents input
            timestep (int): noise scheduler timestep
            encoder_hidden_states (tuple[torch.Tensor, torch.Tensor]): Text encoder hidden states

        Returns:
            torch.Tensor: noise prediction
        """
        prompt_embeds, pooled_prompt_embeds = encoder_hidden_states
        target_size = torch.tensor(
            [
                [self.resolution, self.resolution]
                for _ in range(latent_model_input.size(0))
            ],
            device=latent_model_input.device,
            dtype=torch.float32,
        )
        add_time_ids = torch.cat(
            [target_size, torch.zeros_like(target_size), target_size], dim=1
        )

        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": pooled_prompt_embeds,
        }

        return self.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
        ).sample
