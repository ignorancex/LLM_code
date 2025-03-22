import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torchvision.transforms import functional as TF
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import structural_similarity_index_measure as ssim_metric
from torchmetrics.functional import peak_signal_noise_ratio as psnr_metric


import rawpy
import numpy as np

from improved_consistency_model_conditional import (
    ConsistencySamplingAndEditing,
    ImprovedConsistencyTraining,
    pseudo_huber_loss,
    update_ema_model_,
)



from torch.utils.data import Dataset
from torchvision import transforms as T
import os
from PIL import Image
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

def pack_raw(raw):
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # Normalize to [0,1]
    im = np.expand_dims(im, axis=2)
    H, W = im.shape[0], im.shape[1]
    out = np.concatenate((im[0:H:2,0:W:2,:],
                          im[0:H:2,1:W:2,:],
                          im[1:H:2,1:W:2,:],
                          im[1:H:2,0:W:2,:]), axis=2)
    return out

class SIDDataset(Dataset):
    def __init__(
        self,
        txt_file: str,
        root_dir: str,
        crop_size: Tuple[int, int] = (128, 128),
        resize_size: Tuple[int, int] = (128, 128)
    ):
        self.txt_file = txt_file
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.image_pairs: List[Tuple[str, str]] = []

        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                short_path, long_path, iso, f_number = line.strip().split()
                self.image_pairs.append((short_path, long_path))

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        short_path, long_path = self.image_pairs[idx]
        short_full_path = os.path.join(self.root_dir, short_path)
        long_full_path = os.path.join(self.root_dir, long_path)

        # Extract exposure times from filenames
        short_exposure = float(os.path.basename(short_path).split('_')[-1].replace('s.ARW', ''))
        long_exposure = float(os.path.basename(long_path).split('_')[-1].replace('s.ARW', ''))
        
        # Compute exposure ratio, capped at 300
        ratio = min(long_exposure / short_exposure, 300)

        # Load and postprocess the short RAW image
        with rawpy.imread(short_full_path) as raw:
            short_rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=16
            )
            short_rgb = np.float32(short_rgb) / 65535.0  # Normalize to [0,1]
            short_rgb = short_rgb * ratio               # Apply exposure ratio
            short_rgb = np.clip(short_rgb, 0.0, 1.0)   # Clip to [0,1]

        # Load and postprocess the long RAW (target) image
        with rawpy.imread(long_full_path) as raw:
            long_rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=16
            )
            long_rgb = np.float32(long_rgb) / 65535.0  # Normalize to [0,1]

        # Convert numpy arrays to torch tensors and rearrange dimensions to (C, H, W)
        input_image = torch.from_numpy(short_rgb).permute(2, 0, 1)   # Shape: (3, H, W)
        target_image = torch.from_numpy(long_rgb).permute(2, 0, 1)   # Shape: (3, H, W)

        # Random crop
        _, H, W = input_image.shape
        crop_h, crop_w = self.crop_size
        if H > crop_h and W > crop_w:
            i = np.random.randint(0, H - crop_h + 1)
            j = np.random.randint(0, W - crop_w + 1)
            input_image = input_image[:, i:i + crop_h, j:j + crop_w]
            target_image = target_image[:, i:i + crop_h, j:j + crop_w]

        # Resize to the desired size
        # input_image = TF.resize(input_image, self.resize_size)
        # target_image = TF.resize(target_image, self.resize_size)

        # Normalize images to [-1, 1]
        input_image = (input_image * 2) - 1
        target_image = (target_image * 2) - 1

        return input_image, target_image

# Data module configuration remains the same
@dataclass
class ImageDataModuleConfig:
    data_dir: str = "datasets/sid"  # Path to the dataset directory
    image_size_crop: Tuple[int, int] = (128, 128)  # Size for random cropping
    image_size_resize: Tuple[int, int] = (128, 128)  # Resize to 128x128
    batch_size: int = 34 # Number of images in each batch
    num_workers: int = 28  # Number of worker threads for data loading
    pin_memory: bool = True  # Whether to pin memory in data loader
    persistent_workers: bool = True  # Keep workers alive between epochs

# New SIDDataModule class
class SIDDataModule(LightningDataModule):
    def __init__(self, config: ImageDataModuleConfig) -> None:
        super().__init__()
        self.config = config

    def setup(self, stage: str = None) -> None:
        # Define transforms
        self.transform = T.Lambda(lambda x: (x * 2) - 1)  # Normalize to [-1, 1]

        self.dataset = SIDDataset(
            txt_file=os.path.join(self.config.data_dir, "Sony_train_list.txt"),
            root_dir=self.config.data_dir,
            crop_size=self.config.image_size_crop,
            resize_size=self.config.image_size_resize
        )
        # self.val_dataset = SIDValDataset(
        #         txt_file=os.path.join(self.config.data_dir, "Sony_val_list.txt"),
        #         root_dir=self.config.data_dir,
        #         resize_size=self.config.image_size_resize
        #     )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

def GroupNorm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, channels // 4), num_channels=channels)


class SelfAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.dropout = dropout

        self.qkv_projection = nn.Sequential(
            GroupNorm(in_channels),
            nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, bias=False),
            Rearrange("b (i h d) x y -> i b h (x y) d", i=3, h=n_heads),
        )
        self.output_projection = nn.Sequential(
            Rearrange("b h l d -> b l (h d)"),
            nn.Linear(in_channels, out_channels, bias=False),
            Rearrange("b l d -> b d l"),
            GroupNorm(out_channels),
            nn.Dropout1d(dropout),
        )
        self.residual_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.qkv_projection(x).unbind(dim=0)

        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        output = self.output_projection(output)
        output = rearrange(output, "b c (x y) -> b c x y", x=x.shape[-2], y=x.shape[-1])

        return output + self.residual_projection(x)


class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        noise_level_channels: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Sequential(
            GroupNorm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.Dropout2d(dropout),
        )
        self.noise_level_projection = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(noise_level_channels, out_channels, kernel_size=1),
        )
        self.output_projection = nn.Sequential(
            GroupNorm(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.Dropout2d(dropout),
        )
        self.residual_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor, noise_level: Tensor) -> Tensor:
        h = self.input_projection(x)
        h = h + self.noise_level_projection(noise_level)

        return self.output_projection(h) + self.residual_projection(x)


class UNetBlockWithSelfAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        noise_level_channels: int,
        n_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.unet_block = UNetBlock(
            in_channels, out_channels, noise_level_channels, dropout
        )
        self.self_attention = SelfAttention(
            out_channels, out_channels, n_heads, dropout
        )

    def forward(self, x: Tensor, noise_level: Tensor) -> Tensor:
        return self.self_attention(self.unet_block(x, noise_level))


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            Rearrange("b c (h ph) (w pw) -> b (c ph pw) h w", ph=2, pw=2),
            nn.Conv2d(4 * channels, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class NoiseLevelEmbedding(nn.Module):
    def __init__(self, channels: int, scale: float = 0.02) -> None:
        super().__init__()

        self.W = nn.Parameter(torch.randn(channels // 2) * scale, requires_grad=False)

        self.projection = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.SiLU(),
            nn.Linear(4 * channels, channels),
            Rearrange("b c -> b c () ()"),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = x[:, None] * self.W[None, :] * 2 * torch.pi
        h = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)

        return self.projection(h)


@dataclass
class UNetConfig:
    channels: int = 3
    noise_level_channels: int = 256
    noise_level_scale: float = 0.02
    n_heads: int = 8
    top_blocks_channels: Tuple[int, ...] = (128, 128)
    top_blocks_n_blocks_per_resolution: Tuple[int, ...] = (2, 2)
    top_blocks_has_resampling: Tuple[bool, ...] = (True, True)
    top_blocks_dropout: Tuple[float, ...] = (0.0, 0.0)
    mid_blocks_channels: Tuple[int, ...] = (256, 512)
    mid_blocks_n_blocks_per_resolution: Tuple[int, ...] = (4, 4)
    mid_blocks_has_resampling: Tuple[bool, ...] = (True, False)
    mid_blocks_dropout: Tuple[float, ...] = (0.0, 0.3)


class UNet(nn.Module):
    def __init__(self, config: UNetConfig) -> None:
        super().__init__()

        self.config = config

        self.input_projection = nn.Conv2d(
            config.channels * 2,
            config.top_blocks_channels[0],
            kernel_size=3,
            padding="same",
        )
        self.noise_level_embedding = NoiseLevelEmbedding(
            config.noise_level_channels, config.noise_level_scale
        )
        self.top_encoder_blocks = self._make_encoder_blocks(
            self.config.top_blocks_channels + self.config.mid_blocks_channels[:1],
            self.config.top_blocks_n_blocks_per_resolution,
            self.config.top_blocks_has_resampling,
            self.config.top_blocks_dropout,
            self._make_top_block,
        )
        self.mid_encoder_blocks = self._make_encoder_blocks(
            self.config.mid_blocks_channels + self.config.mid_blocks_channels[-1:],
            self.config.mid_blocks_n_blocks_per_resolution,
            self.config.mid_blocks_has_resampling,
            self.config.mid_blocks_dropout,
            self._make_mid_block,
        )
        self.mid_decoder_blocks = self._make_decoder_blocks(
            self.config.mid_blocks_channels + self.config.mid_blocks_channels[-1:],
            self.config.mid_blocks_n_blocks_per_resolution,
            self.config.mid_blocks_has_resampling,
            self.config.mid_blocks_dropout,
            self._make_mid_block,
        )
        self.top_decoder_blocks = self._make_decoder_blocks(
            self.config.top_blocks_channels + self.config.mid_blocks_channels[:1],
            self.config.top_blocks_n_blocks_per_resolution,
            self.config.top_blocks_has_resampling,
            self.config.top_blocks_dropout,
            self._make_top_block,
        )
        self.output_projection = nn.Conv2d(
            config.top_blocks_channels[0],
            config.channels,
            kernel_size=3,
            padding="same",
        )

    def forward(self, x: Tensor, noise_level: Tensor, v: Tensor) -> Tensor:
        x = torch.cat([x, v], dim = 1)
        h = self.input_projection(x)
        noise_level = self.noise_level_embedding(noise_level)

        top_encoder_embeddings = []
        for block in self.top_encoder_blocks:
            if isinstance(block, UNetBlock):
                h = block(h, noise_level)
                top_encoder_embeddings.append(h)
            else:
                h = block(h)

        mid_encoder_embeddings = []
        for block in self.mid_encoder_blocks:
            if isinstance(block, UNetBlockWithSelfAttention):
                h = block(h, noise_level)
                mid_encoder_embeddings.append(h)
            else:
                h = block(h)

        for block in self.mid_decoder_blocks:
            if isinstance(block, UNetBlockWithSelfAttention):
                h = torch.cat((h, mid_encoder_embeddings.pop()), dim=1)
                h = block(h, noise_level)
            else:
                h = block(h)

        for block in self.top_decoder_blocks:
            if isinstance(block, UNetBlock):
                h = torch.cat((h, top_encoder_embeddings.pop()), dim=1)
                h = block(h, noise_level)
            else:
                h = block(h)

        output = self.output_projection(h)

        # Concatenate the infrared output with a 3-channel tensor of zeros
        # zero_channels = torch.zeros_like(output)

        # return torch.cat([output, zero_channels], dim=1)
        return output

    def _make_encoder_blocks(
        self,
        channels: Tuple[int, ...],
        n_blocks_per_resolution: Tuple[int, ...],
        has_resampling: Tuple[bool, ...],
        dropout: Tuple[float, ...],
        block_fn: Callable[[], nn.Module],
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()

        channel_pairs = list(zip(channels[:-1], channels[1:]))
        for idx, (in_channels, out_channels) in enumerate(channel_pairs):
            for _ in range(n_blocks_per_resolution[idx]):
                blocks.append(block_fn(in_channels, out_channels, dropout[idx]))
                in_channels = out_channels

            if has_resampling[idx]:
                blocks.append(Downsample(out_channels))

        return blocks

    def _make_decoder_blocks(
        self,
        channels: Tuple[int, ...],
        n_blocks_per_resolution: Tuple[int, ...],
        has_resampling: Tuple[bool, ...],
        dropout: Tuple[float, ...],
        block_fn: Callable[[], nn.Module],
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()

        channel_pairs = list(zip(channels[:-1], channels[1:]))[::-1]
        for idx, (out_channels, in_channels) in enumerate(channel_pairs):
            if has_resampling[::-1][idx]:
                blocks.append(Upsample(in_channels))

            inner_blocks = []
            for _ in range(n_blocks_per_resolution[::-1][idx]):
                inner_blocks.append(
                    block_fn(in_channels * 2, out_channels, dropout[::-1][idx])
                )
                out_channels = in_channels
            blocks.extend(inner_blocks[::-1])

        return blocks

    def _make_top_block(
        self, in_channels: int, out_channels: int, dropout: float
    ) -> UNetBlock:
        return UNetBlock(
            in_channels,
            out_channels,
            self.config.noise_level_channels,
            dropout,
        )

    def _make_mid_block(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
    ) -> UNetBlockWithSelfAttention:
        return UNetBlockWithSelfAttention(
            in_channels,
            out_channels,
            self.config.noise_level_channels,
            self.config.n_heads,
            dropout,
        )

    def save_pretrained(self, pretrained_path: str) -> None:
        os.makedirs(pretrained_path, exist_ok=True)

        with open(os.path.join(pretrained_path, "config.json"), mode="w") as f:
            json.dump(asdict(self.config), f)

        torch.save(self.state_dict(), os.path.join(pretrained_path, "model.pt"))

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "UNet":
        with open(os.path.join(pretrained_path, "config.json"), mode="r") as f:
            config_dict = json.load(f)
        config = UNetConfig(**config_dict)

        model = cls(config)

        state_dict = torch.load(
            os.path.join(pretrained_path, "model.pt"), map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict)

        return model


# summary(UNet(UNetConfig()), input_size=((1, 6, 32, 32), (1,)))


@dataclass
class LitImprovedConsistencyModelConfig:
    ema_decay_rate: float = 0.99993
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.995)
    lr_scheduler_start_factor: float = 1e-5
    lr_scheduler_iters: int = 10_000
    sample_every_n_steps: int = 10_000
    num_samples: int = 8
    sampling_sigmas: Tuple[Tuple[int, ...], ...] = (
        (80,),
        (80.0, 0.661),
        (80.0, 24.4, 5.84, 0.9, 0.661),
    )


class LitImprovedConsistencyModel(LightningModule):
    def __init__(
        self,
        consistency_training: ImprovedConsistencyTraining,
        consistency_sampling: ConsistencySamplingAndEditing,
        model: UNet,
        ema_model: UNet,
        config: LitImprovedConsistencyModelConfig,
    ) -> None:
        super().__init__()

        self.consistency_training = consistency_training
        self.consistency_sampling = consistency_sampling
        self.model = model
        self.ema_model = ema_model
        self.config = config

        # Freeze the EMA model and set it to eval mode
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.ema_model = self.ema_model.eval()

    def training_step(self, batch: Union[Tensor, List[Tensor]], batch_idx: int) -> None:
        # if isinstance(batch, list):
        #     batch = batch[0]

        visible_images, infrared_images = batch  # Unpack the batch

        output = self.consistency_training(
            self.model, 
            infrared_images, 
            visible_images,  # Pass visible images to the training function
            self.global_step, 
            self.trainer.max_steps
        )

        loss = (
            pseudo_huber_loss(output.predicted, output.target) * output.loss_weights
        ).mean()

        self.log_dict({"train_loss": loss, "num_timesteps": output.num_timesteps})

        return loss

    def on_train_batch_end(
        self, outputs: Any, batch: Union[Tensor, List[Tensor]], batch_idx: int
    ) -> None:
        update_ema_model_(self.model, self.ema_model, self.config.ema_decay_rate)

        if (
            (self.global_step + 1) % self.config.sample_every_n_steps == 0
        ) or self.global_step == 0:
            self.__sample_and_log_samples(batch)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr, betas=self.config.betas
        )
        sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=self.config.lr_scheduler_start_factor,
            total_iters=self.config.lr_scheduler_iters,
        )
        sched = {"scheduler": sched, "interval": "step", "frequency": 1}

        return [opt], [sched]

    @torch.no_grad()
    def __sample_and_log_samples(self, batch: Union[Tensor, List[Tensor]]) -> None:
        if isinstance(batch, list):
            batch = batch[0]

        # Ensure the number of samples does not exceed the batch size
        num_samples = min(self.config.num_samples, batch.shape[0])
        noise = torch.randn_like(batch[:num_samples])

        # Log ground truth samples
        self.__log_images(
            batch[:num_samples].detach().clone(), f"ground_truth", self.global_step
        )

        for sigmas in self.config.sampling_sigmas:
            samples = self.consistency_sampling(
                self.ema_model, noise, sigmas, clip_denoised=True, verbose=True
            )
            samples = samples.clamp(min=-1.0, max=1.0)

            # Generated samples
            self.__log_images(
                samples,
                f"generated_samples-sigmas={sigmas}",
                self.global_step,
            )

    @torch.no_grad()
    def __log_images(self, images: Tensor, title: str, global_step: int) -> None:
        images = images.detach().float()

        grid = make_grid(
            images.clamp(-1.0, 1.0), value_range=(-1.0, 1.0), normalize=True
        )
        self.logger.experiment.add_image(title, grid, global_step)
    
    def on_train_epoch_end(self) -> None:
        # Retrieve the loss logged during the epoch
        train_loss = self.trainer.callback_metrics.get("train_loss", None)
        
        if train_loss is not None:
            print(f"Epoch {self.current_epoch} - Train Loss: {train_loss:.4f}")


@dataclass
class TrainingConfig:
    image_dm_config: ImageDataModuleConfig
    unet_config: UNetConfig
    consistency_training: ImprovedConsistencyTraining
    consistency_sampling: ConsistencySamplingAndEditing
    lit_icm_config: LitImprovedConsistencyModelConfig
    trainer: Trainer
    model_ckpt_path: str = "checkpoints/sid"
    seed: int = 42

def run_training(config: TrainingConfig) -> None:
    # Set seed
    seed_everything(config.seed)

    # Create data module
    dm = SIDDataModule(config.image_dm_config)
    dm.setup()
    print("DataModule setup complete.")

    # Create model and its EMA
    model = UNet(config.unet_config)
    ema_model = UNet(config.unet_config)
    ema_model.load_state_dict(model.state_dict())

    # Create lightning module
    lit_icm = LitImprovedConsistencyModel(
        config.consistency_training,
        config.consistency_sampling,
        model,
        ema_model,
        config.lit_icm_config,
    )

    print("Lightning module created.")

    # Run training
    print("Starting training...")
    config.trainer.fit(lit_icm, datamodule=dm)
    print("Training completed.")

    # Save model
    lit_icm.model.save_pretrained(config.model_ckpt_path)
    print("Model saved.")

def main():
    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_sid",
        filename="{epoch}-{step}",
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=20,  # Adjust as needed
    )

    # Set up the logger
    logger = TensorBoardLogger("logs", name="sid")

    training_config = TrainingConfig(
        image_dm_config=ImageDataModuleConfig(data_dir="../datasets/sid"),
        unet_config=UNetConfig(),
        consistency_training=ImprovedConsistencyTraining(final_timesteps=11),
        consistency_sampling=ConsistencySamplingAndEditing(),
        lit_icm_config=LitImprovedConsistencyModelConfig(
            sample_every_n_steps=2100000, lr_scheduler_iters=1000
        ),
        trainer=Trainer(
            max_steps=100,
            precision="16",
            log_every_n_steps=10,
            logger=logger,
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
                checkpoint_callback,  # Add the checkpoint callback here
            ],
        ),
    )
    run_training(training_config)

if __name__ == "__main__":
    main()