from pathlib import Path

import torch
from PIL import Image
from torch.cuda.amp import autocast
from tqdm.auto import tqdm

from src.models import StableDiffusion
from src.trainer.base_trainer import BaseTrainer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import get_image_name_by_index


class InferenceCombinedGeneration(BaseTrainer):
    """
    InferenceGeneration:

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model: StableDiffusion,
        original_model: StableDiffusion,
        config,
        device,
        dataloaders,
        batch_transforms=None,
    ):
        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.original_model = original_model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        self.start_timestep_index = None
        self.global_image_index = 0
        if config.inferencer.get("from_pretrained"):
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        for start_timestep_index in self.cfg_trainer.start_timestep_indexs:
            self.global_image_index = 0

            self.start_timestep_index = start_timestep_index
            for part, dataloader in self.evaluation_dataloaders.items():
                self._inference_part(
                    part,
                    dataloader,
                )

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()
        set_random_seed(self.cfg_trainer.seed)

        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                for batch in tqdm(
                    dataloader,
                    desc=part,
                    total=len(dataloader),
                ):
                    batch = self.move_batch_to_device(batch)
                    batch = self.transform_batch(batch)

                    images = self._get_images(batch)

                    self._save_images(images)

    def _get_images(self, batch: dict[str, torch.Tensor]) -> list[Image]:
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

        _, pil_images = self.model.sample_image_inference(
            latents=latents,
            start_timestep_index=self.start_timestep_index,
            end_timestep_index=self.cfg_trainer.end_timestep_index,
            batch=batch,
            do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
        )
        return pil_images

    def _save_images(self, images: list[Image]) -> None:
        image_path = Path(
            self.cfg_trainer.save_images_path + f"_{self.start_timestep_index}"
        )
        image_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(image_path / get_image_name_by_index(self.global_image_index))
            self.global_image_index += 1
