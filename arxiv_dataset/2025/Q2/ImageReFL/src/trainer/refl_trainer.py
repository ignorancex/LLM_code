import random
from contextlib import nullcontext

import torch

from src.constants.dataset import DatasetColumns
from src.trainer.base_trainer import BaseTrainer


class ReFLTrainer(BaseTrainer):
    """
    Trainer class.
    Reproduce ReFL training.
    """

    def _sample_image_train(self, batch: dict[str, torch.Tensor]):
        self.model.set_timesteps(self.cfg_trainer.max_mid_timestep, device=self.device)

        mid_timestep = (
            random.randint(
                self.cfg_trainer.min_mid_timestep,
                self.cfg_trainer.max_mid_timestep - 1,
            )
            if self.is_train
            else self.cfg_trainer.max_mid_timestep - 1
        )

        with torch.no_grad():
            latents, encoder_hidden_states = self.model.do_k_diffusion_steps(
                latents=None,
                start_timestep_index=0,
                end_timestep_index=mid_timestep,
                batch=batch,
                return_pred_original=False,
                do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
            )

        batch["image"] = self.model.sample_image(
            latents=latents,
            start_timestep_index=mid_timestep,
            end_timestep_index=mid_timestep + 1,
            encoder_hidden_states=encoder_hidden_states,
            batch=batch,
            do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
            detach_main_path=self.cfg_trainer.detach_main_path,
        )

        if DatasetColumns.original_image.name in batch:
            context_manager = (
                nullcontext()
                if self.cfg_trainer.mse_loss_scale > 0
                else torch.no_grad()
            )
            with context_manager:
                noised_latents, noise = self.model.get_noisy_latents_from_images(
                    images=batch[DatasetColumns.original_image.name],
                    timestep_index=mid_timestep,
                )
                if self.cfg_trainer.do_classifier_free_guidance:
                    encoder_hidden_states = self.model.get_encoder_hidden_states(
                        batch=batch, do_classifier_free_guidance=False
                    )
                _, noise_pred = self.model.predict_next_latents(
                    latents=noised_latents,
                    timestep_index=mid_timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    batch=batch,
                )
                mse_loss = torch.nn.functional.mse_loss(noise_pred, noise)
                batch["mse_loss"] = mse_loss.detach()

            if self.cfg_trainer.mse_loss_scale > 0:
                batch["loss"] = mse_loss * self.cfg_trainer.mse_loss_scale

    def _sample_image_eval(self, batch: dict[str, torch.Tensor]):
        self.model.set_timesteps(self.cfg_trainer.max_mid_timestep, device=self.device)

        mid_timestep = (
            random.randint(
                self.cfg_trainer.min_mid_timestep,
                self.cfg_trainer.max_mid_timestep - 1,
            )
            if self.is_train
            else self.cfg_trainer.max_mid_timestep - 1
        )

        with torch.no_grad():
            latents, encoder_hidden_states = self.model.do_k_diffusion_steps(
                latents=None,
                start_timestep_index=0,
                end_timestep_index=mid_timestep,
                batch=batch,
                return_pred_original=False,
                do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
            )

        batch["image"] = self.model.sample_image(
            latents=latents,
            start_timestep_index=mid_timestep,
            end_timestep_index=mid_timestep + 1,
            encoder_hidden_states=encoder_hidden_states,
            batch=batch,
            do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
            detach_main_path=self.cfg_trainer.detach_main_path,
        )
