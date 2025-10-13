import torch

from src.metrics.tracker import MetricTracker
from src.models import StableDiffusion
from src.reward_models import BaseModel
from src.trainer.inferencer import Inferencer


class CombinedGenerationInferencer(Inferencer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model: StableDiffusion,
        original_model: StableDiffusion,
        reward_models: list[BaseModel],
        config,
        device,
        dataloaders,
        writer,
        batch_transforms=None,
    ):
        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.writer = writer

        self.device = device

        self.model = model
        self.original_model = original_model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        self.reward_models = reward_models

        # define metrics
        self.loss_names = [
            reward_model.model_suffix for reward_model in self.reward_models
        ]
        self.all_metrics = MetricTracker(
            *self.loss_names,
            writer=self.writer,
        )
        self.start_timestep_index = None
        self.global_image_index = 0
        if config.inferencer.get("from_pretrained"):
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def _sample_image(self, batch: dict[str, torch.Tensor]):
        self.model.set_timesteps(
            self.cfg_trainer.end_timestep_index, device=self.device
        )
        self.original_model.set_timesteps(
            self.cfg_trainer.end_timestep_index, device=self.device
        )

        latents, _ = self.original_model.do_k_diffusion_steps(
            latents=None,
            start_timestep_index=0,
            end_timestep_index=self.start_timestep_index,
            batch=batch,
            do_classifier_free_guidance=True,
        )

        reward_images, pil_images = self.model.sample_image_inference(
            latents=latents,
            start_timestep_index=self.start_timestep_index,
            end_timestep_index=self.cfg_trainer.end_timestep_index,
            batch=batch,
            do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
        )
        batch["image"] = reward_images
        batch["pil_images"] = pil_images
