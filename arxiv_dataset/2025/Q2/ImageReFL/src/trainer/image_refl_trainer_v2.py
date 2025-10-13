import torch

from src.constants.dataset import DatasetColumns
from src.reward_models import ClipScore
from src.trainer.image_refl_trainer import ImageReFLTrainer


class ImageReFLTrainerV2(ImageReFLTrainer):
    """
    Trainer class.
    Reproduce ImageReFL training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_model = ClipScore(device=self.config.trainer.device).to(
            self.config.trainer.device
        )

    def _sample_image_train(self, batch: dict[str, torch.Tensor]) -> None:
        super()._sample_image_train(batch)
        if "loss" not in batch:
            batch["loss"] = 0.0
        encoded_image = self.clip_model.model.encode_image(batch["image"])
        encoded_image = encoded_image / encoded_image.norm(dim=1, keepdim=True)
        encoded_original_image = self.clip_model.model.encode_image(
            self.model.get_reward_image(batch["original_image"])
        )
        encoded_original_image = encoded_original_image / encoded_original_image.norm(
            dim=1, keepdim=True
        )
        score = (1 - torch.sum(encoded_image * encoded_original_image, dim=1)).mean()

        batch["loss"] += self.cfg_trainer.clip_scale * score
        batch["cosine_similarity"] = score.detach()

    def _sample_image_eval(self, batch: dict[str, torch.Tensor]):
        self.model.set_timesteps(self.cfg_trainer.max_mid_timestep, device=self.device)

        original_images = batch[DatasetColumns.original_image.name]

        latents, _ = self.model.get_noisy_latents_from_images(
            images=original_images,
            timestep_index=self.cfg_trainer.start_timestep_index,
        )

        batch["image"] = self.model.sample_image(
            latents=latents,
            start_timestep_index=self.cfg_trainer.start_timestep_index,
            end_timestep_index=self.cfg_trainer.max_mid_timestep,
            batch=batch,
        )

    def _get_train_loss_names(self):
        train_loss_names = super()._get_train_loss_names()
        train_loss_names.append("cosine_similarity")
        return train_loss_names
