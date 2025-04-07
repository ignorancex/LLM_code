from typing import Union

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, PNDMScheduler, UNet2DConditionModel, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel

from models.losses import Loss

NoiseScheduler = Union[DDPMScheduler, DDIMScheduler, PNDMScheduler]


class Trainer:
    def __init__(self, accelerator: Accelerator, project_name: str):
        self.accelerator = accelerator
        self.project_name = project_name

    def fit(
        self,
        model: UNet2DModel,
        noise_scheduler: NoiseScheduler,
        criterion: Loss,
        data_loader: DataLoader,
        num_train_timesteps: int = 1000,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        num_warmup_steps: int = 500,
    ):
        self.accelerator.init_trackers(self.project_name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # type: ignore
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=(len(data_loader) * num_epochs),
        )

        model = self.accelerator.prepare_model(model)
        optimizer = self.accelerator.prepare_optimizer(optimizer)
        data_loader = self.accelerator.prepare_data_loader(data_loader)
        lr_scheduler = self.accelerator.prepare_scheduler(lr_scheduler)  # type: ignore

        def compute_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            t = torch.randint(
                0,
                num_train_timesteps,
                (x.shape[0],),
                device=self.accelerator.device,
                dtype=torch.int64,
            )

            noise = torch.randn_like(x)
            x_t = noise_scheduler.add_noise(  # type: ignore
                original_samples=x,
                noise=noise,
                timesteps=t,
            )

            noise_t = model(x_t, t).sample  # type: ignore
            output = torch.norm((noise_t - noise).view(x.shape[0], -1), dim=1) ** 2
            return criterion(output=output, target=y)  # type: ignore

        for epoch in range(num_epochs):
            train_progress_bar = tqdm(data_loader)
            train_progress_bar.set_description(f"Epoch {epoch}")
            mean_train_loss: float = 0
            for batch in train_progress_bar:
                with self.accelerator.accumulate(model):
                    optimizer.zero_grad()
                    train_loss = compute_loss(x=batch[0], y=batch[1])
                    mean_train_loss += train_loss.detach().item() / len(data_loader)

                    self.accelerator.backward(train_loss)
                    self.accelerator.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
                    optimizer.step()
                    lr_scheduler.step()

            # logging
            logs = {"train_loss": mean_train_loss}
            self.accelerator.log(logs, step=epoch)

        self.accelerator.end_training()
        self.accelerator.unwrap_model(model)


class StableDiffusionTrainer:
    def __init__(self, accelerator: Accelerator, project_name: str):
        self.accelerator = accelerator
        self.project_name = project_name

    def fit(
        self,
        model: UNet2DConditionModel,
        noise_scheduler: NoiseScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        criterion: Loss,
        data_loader: DataLoader,
        num_train_timesteps: int = 1000,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        num_warmup_steps: int = 500,
    ):
        self.accelerator.init_trackers(self.project_name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # type: ignore
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=(len(data_loader) * num_epochs),
        )

        model = self.accelerator.prepare_model(model)
        optimizer = self.accelerator.prepare_optimizer(optimizer)
        data_loader = self.accelerator.prepare_data_loader(data_loader)
        lr_scheduler = self.accelerator.prepare_scheduler(lr_scheduler)  # type: ignore

        vae = self.accelerator.prepare_model(vae)
        text_encoder = self.accelerator.prepare_model(text_encoder)

        def compute_loss(
            image: torch.Tensor, label: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            latents = vae.encode(image).latent_dist.sample()  # type: ignore
            latents = latents * 0.18215
            encoder_hidden_states = text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state

            t = torch.randint(
                0,
                num_train_timesteps,
                (latents.shape[0],),
                device=self.accelerator.device,
                dtype=torch.int64,
            )
            noise = torch.randn_like(latents)

            latents_t = noise_scheduler.add_noise(  # type: ignore
                original_samples=latents,
                noise=noise,
                timesteps=t,
            )

            noise_t = model(latents_t, t, encoder_hidden_states).sample  # type: ignore

            output = torch.norm((noise_t - noise).view(latents.shape[0], -1), dim=1) ** 2
            return criterion(output=output, target=label)  # type: ignore

        for epoch in range(num_epochs):
            train_progress_bar = tqdm(data_loader)
            train_progress_bar.set_description(f"Epoch {epoch}")
            mean_train_loss: float = 0
            for batch in train_progress_bar:
                with self.accelerator.accumulate(model):
                    with torch.autocast("cuda"):
                        optimizer.zero_grad()
                        train_loss = compute_loss(
                            image=batch["image"],
                            label=batch["label"],
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )
                        mean_train_loss += train_loss.detach().item() / len(data_loader)

                        self.accelerator.backward(train_loss)
                        self.accelerator.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
                        optimizer.step()
                        lr_scheduler.step()

            # logging
            logs = {"train_loss": mean_train_loss}
            self.accelerator.log(logs, step=epoch)

        self.accelerator.end_training()
        self.accelerator.unwrap_model(model)
        self.accelerator.unwrap_model(vae)
        self.accelerator.unwrap_model(text_encoder)
