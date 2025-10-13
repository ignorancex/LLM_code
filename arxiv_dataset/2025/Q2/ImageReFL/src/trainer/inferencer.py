import typing as tp
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from tqdm.auto import tqdm

from src.constants.dataset import DatasetColumns
from src.metrics.tracker import MetricTracker
from src.models import StableDiffusion
from src.reward_models import BaseModel
from src.trainer.base_trainer import BaseTrainer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import get_image_name_by_index


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model: StableDiffusion,
        reward_models: list[BaseModel],
        config,
        device,
        dataloaders,
        writer,
        batch_transforms=None,
        **kwargs,
    ):
        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.writer = writer

        self.device = device

        self.model = model
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

    def run_inference(self):
        part_logs = {}

        for start_timestep_index in self.cfg_trainer.start_timestep_indexs:
            self.global_image_index = 0

            self.start_timestep_index = start_timestep_index
            for part, dataloader in self.evaluation_dataloaders.items():
                logs = self._inference_part(
                    part,
                    dataloader,
                )
                part_logs[part] = logs

            self.writer.set_step(start_timestep_index)
            for metric_name in self.all_metrics.keys():
                self.writer.add_scalar(
                    f"{metric_name}", self.all_metrics.avg(metric_name)
                )
            self.all_metrics.reset()
        return part_logs

    def _sample_image(self, batch: dict[str, torch.Tensor]):
        self.model.set_timesteps(
            self.cfg_trainer.end_timestep_index, device=self.device
        )

        original_images = batch[DatasetColumns.original_image.name]

        noised_latents, _ = self.model.get_noisy_latents_from_images(
            images=original_images,
            timestep_index=self.start_timestep_index,
        )

        reward_images, pil_images = self.model.sample_image_inference(
            latents=noised_latents,
            start_timestep_index=self.start_timestep_index,
            end_timestep_index=self.cfg_trainer.end_timestep_index,
            batch=batch,
            do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
        )
        batch["image"] = reward_images
        batch["pil_images"] = pil_images

    def process_batch(
        self, batch: dict[str, torch.Tensor], metrics: MetricTracker
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        self._sample_image(batch)

        for reward_model in self.reward_models:
            reward_model.score(batch=batch)

        return batch

    def _save_images(self, batch: dict[str, tp.Any]) -> None:
        images = batch["pil_images"]
        image_path = Path(
            self.cfg_trainer.save_images_path + f"_{self.start_timestep_index}"
        )
        image_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(image_path / get_image_name_by_index(self.global_image_index))
            self.global_image_index += 1

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.all_metrics.reset()
        set_random_seed(self.cfg_trainer.seed)

        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                for batch in tqdm(
                    dataloader,
                    desc=part,
                    total=len(dataloader),
                ):
                    batch = self.process_batch(batch=batch, metrics=self.all_metrics)

                    for loss_name in self.loss_names:
                        if loss_name in batch:
                            self.all_metrics.update(loss_name, batch[loss_name].item())
                    if self.cfg_trainer.save_images_path:
                        self._save_images(batch)

        self.writer.add_image(
            image_name="generated",
            image=batch["image"],
        )
        return self.all_metrics.result()
