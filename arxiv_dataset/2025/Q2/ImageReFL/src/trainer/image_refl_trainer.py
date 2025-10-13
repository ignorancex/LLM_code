import random
from contextlib import nullcontext

import torch

from src.constants.dataset import DatasetColumns
from src.trainer.refl_trainer import ReFLTrainer


class ImageReFLTrainer(ReFLTrainer):
    """
    Trainer class.
    Reproduce ImageReFL training.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.cfg_trainer.min_mid_timestep <= self.cfg_trainer.max_mid_timestep
        assert (
            self.cfg_trainer.start_timestep_index <= self.cfg_trainer.max_mid_timestep
        )
        assert (
            self.cfg_trainer.start_timestep_index <= self.cfg_trainer.min_mid_timestep
        )

        self.state = 0

    def _sample_image_train(self, batch: dict[str, torch.Tensor]) -> None:
        if self.cfg_trainer.state_del is not None:
            self.state += 1
            self.state %= self.cfg_trainer.state_del

        if (
            self.cfg_trainer.state_del is not None
            and self.state % self.cfg_trainer.state_del == 0
        ):
            super()._sample_image_train(batch=batch)
            return

        self.model.set_timesteps(self.cfg_trainer.max_mid_timestep, device=self.device)

        mid_timestep = (
            random.randint(
                self.cfg_trainer.min_mid_timestep,
                self.cfg_trainer.max_mid_timestep - 1,
            )
            if self.is_train
            else self.cfg_trainer.max_mid_timestep - 1
        )

        original_images = batch[DatasetColumns.original_image.name]

        with torch.no_grad():
            noised_latents, noise = self.model.get_noisy_latents_from_images(
                images=original_images,
                timestep_index=self.cfg_trainer.start_timestep_index,
            )
            latents, encoder_hidden_states = self.model.do_k_diffusion_steps(
                latents=noised_latents,
                start_timestep_index=self.cfg_trainer.start_timestep_index,
                end_timestep_index=mid_timestep,
                batch=batch,
                return_pred_original=False,
            )

        batch["image"] = self.model.sample_image(
            latents=latents,
            start_timestep_index=mid_timestep,
            end_timestep_index=mid_timestep + 1,
            encoder_hidden_states=encoder_hidden_states,
            batch=batch,
        )

        context_manager = (
            nullcontext() if self.cfg_trainer.mse_loss_scale > 0 else torch.no_grad()
        )
        with context_manager:
            _, noise_pred = self.model.predict_next_latents(
                latents=noised_latents,
                timestep_index=self.cfg_trainer.start_timestep_index,
                encoder_hidden_states=encoder_hidden_states,
                batch=batch,
            )
            mse_loss = torch.nn.functional.mse_loss(noise_pred, noise)
            batch["mse_loss"] = mse_loss.detach()

        if self.cfg_trainer.mse_loss_scale > 0:
            batch["loss"] = mse_loss * self.cfg_trainer.mse_loss_scale

    def _get_train_loss_names(self):
        train_loss_names = super()._get_train_loss_names()
        train_loss_names.append("mse_loss")
        return train_loss_names
