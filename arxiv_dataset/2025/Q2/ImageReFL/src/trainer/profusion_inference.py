import torch
from diffusers import DDIMScheduler

from src.constants.dataset import DatasetColumns
from src.trainer.combined_generation_inference import CombinedGenerationInferencer


class ProFusionInferencer(CombinedGenerationInferencer):
    """
    Inferencer with Profusion.
    """

    def _sample_image(self, batch: dict[str, torch.Tensor]):
        self.model.set_timesteps(
            self.cfg_trainer.end_timestep_index, device=self.device
        )
        self.original_model.set_timesteps(
            self.cfg_trainer.end_timestep_index, device=self.device
        )
        T = self.cfg_trainer.end_timestep_index
        self.model.set_timesteps(T, device=self.device)
        self.original_model.set_timesteps(T, device=self.device)

        encoder_hidden_states = self.original_model.get_encoder_hidden_states(
            batch=batch, do_classifier_free_guidance=False
        )
        encoder_hidden_states_cfg = self.original_model.get_encoder_hidden_states(
            batch=batch, do_classifier_free_guidance=True
        )

        latents, _ = self.original_model.do_k_diffusion_steps(
            latents=None,
            start_timestep_index=0,
            end_timestep_index=self.start_timestep_index,
            batch=batch,
            do_classifier_free_guidance=True,
        )

        for timestep_index in range(
            self.start_timestep_index, len(self.original_model.timesteps)
        ):
            noise_pred = self.model.get_noise_prediction(
                latents=latents,
                timestep_index=timestep_index,
                encoder_hidden_states=(
                    encoder_hidden_states_cfg
                    if self.cfg_trainer.do_classifier_free_guidance
                    else encoder_hidden_states
                ),
                do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
            )
            timestep = self.original_model.timesteps[timestep_index]
            prev_timestep = (
                self.original_model.timesteps[timestep_index + 1]
                if timestep_index < len(self.original_model.timesteps) - 1
                else 0
            )

            alpha_prod_t = self.original_model.noise_scheduler.alphas_cumprod[timestep]
            alpha_prod_t_prev = (
                self.original_model.noise_scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.original_model.noise_scheduler.final_alpha_cumprod
            )

            variance = self.original_model.noise_scheduler._get_variance(
                timestep, prev_timestep
            )
            sigma_t = variance.sqrt()
            noise = torch.randn_like(latents)

            latents = (
                latents
                - noise_pred
                * (sigma_t**2 * torch.sqrt(1 - alpha_prod_t))
                / (1 - alpha_prod_t_prev)
                + sigma_t
                * noise
                * torch.sqrt(
                    (1 - alpha_prod_t) * (2 - 2 * alpha_prod_t_prev - sigma_t**2)
                )
                / (1 - alpha_prod_t_prev)
            )

            if self.cfg_trainer.with_two_models_cfg_prediction:
                latent_model_input = (
                    self.original_model.noise_scheduler.scale_model_input(
                        sample=torch.cat([latents] * 2),
                        timestep=timestep,
                    )
                )
                original_noise_pred_uncond, original_noise_pred_text = (
                    self.original_model.unet(
                        latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=encoder_hidden_states_cfg,
                    ).sample.chunk(2)
                )

                finetuned_noise_pred_uncond, finetuned_noise_pred_text = (
                    self.model.unet(
                        latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=encoder_hidden_states_cfg,
                    ).sample.chunk(2)
                )

                noise_pred = (
                    finetuned_noise_pred_uncond
                    + +self.model.guidance_scale
                    * (finetuned_noise_pred_text - finetuned_noise_pred_uncond)
                    + self.original_model.guidance_scale
                    * (original_noise_pred_text - original_noise_pred_uncond)
                )

            else:
                noise_pred = self.model.get_noise_prediction(
                    latents=latents,
                    timestep_index=timestep_index,
                    encoder_hidden_states=encoder_hidden_states_cfg,
                    do_classifier_free_guidance=True,
                    detach_main_path=False,
                )

            latents = self.model.sample_next_latents(
                latents=latents,
                noise_pred=noise_pred,
                timestep_index=timestep_index,
                return_pred_original=False,
            )

        latents /= self.model.vae.config.scaling_factor

        raw_image = self.model.vae.decode(latents).sample
        reward_images, pil_images = self.model.get_reward_image(
            raw_image
        ), self.model.get_pil_image(raw_image)

        batch["image"] = reward_images
        batch["pil_images"] = pil_images
