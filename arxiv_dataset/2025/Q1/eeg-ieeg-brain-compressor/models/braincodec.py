from typing import Dict, Union

import torch
from deepspeed.ops.adam import FusedAdam
from einops import rearrange
from torchmetrics import MeanMetric
from transformers import get_cosine_schedule_with_warmup

from utils.balancer import Balancer
from utils.loss import discriminator_loss, generator_loss

from . import EEGCodec
from .discriminators import MultiResolutionDiscriminator


class BrainCodec(EEGCodec):
    def __init__(
        self,
        model: torch.nn.Module,
        quantizer: torch.nn.Module,
        sample_rate: int = 512,
        weights: Dict[str, float] = {
            "lambda_r": 1.0,
            "lambda_l": 0.0,
            "lambda_s": 0.0,
            "lambda_f": 0.0,
            "lambda_a": 0.0,
            "lambda_q": 1.0,
        },
        use_balancer: bool = False,
        train_discriminator=False,
        lr: Union[float, Dict[str, float]] = 0.1,
        warmup=0,
        training_steps=1000,
        accumulate_grad_batches=None,
        load_model=None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model", "quantizer"])

        self.model = model
        self.quantizer = quantizer
        self.sample_rate = sample_rate
        self.use_balancer = use_balancer
        self.lr = lr
        self.warmup = warmup
        self.training_steps = training_steps
        self.train_discriminator = train_discriminator

        self.train_prd = MeanMetric()
        self.test_prd = MeanMetric()

        self.accumulate_grad_batches = (
            accumulate_grad_batches if accumulate_grad_batches is not None else 1
        )

        self.balancer = Balancer(
            weights={
                "reconstruction_loss": weights["lambda_r"],
                "line_loss": weights["lambda_l"],
                "spectral_loss": weights["lambda_s"],
                "feature_loss": weights["lambda_f"],
                "adversarial_loss": weights["lambda_a"],
            },
        )
        self.discriminator = MultiResolutionDiscriminator(
            resolutions=[
                (2048, 512, 2048),
                (1024, 256, 1024),
                (512, 128, 512),
                (256, 64, 256),
                (128, 32, 128),
            ]
        )

        self.automatic_optimization = False

        if load_model is not None:
            weights = torch.load(load_model, weights_only=True)
            self.load_state_dict(weights, strict=False)

    def forward(self, audio_input):
        output, _ = self._common_step(audio_input)
        return output

    def _common_step(self, audio_input):
        audio_input = rearrange(
            audio_input,
            "batch channel length -> (batch channel) 1 length",
        )
        features = self.model.encode(audio_input)
        quantized, _, commit_loss = self.quantizer(features[0])
        output = self.model.decode(quantized)

        return output, commit_loss

    def training_step(self, batch, batch_idx):
        x = batch.data

        optimizer_d, optimizer_g = self.optimizers()
        scheduler_d, scheduler_g = self.lr_schedulers()

        audio_hat, commit_loss = self._common_step(x)

        x = rearrange(
            x,
            "batch channel length -> (batch channel) 1 length",
        )

        is_last_batch_to_accumulate = (
            batch_idx + 1
        ) % self.accumulate_grad_batches == 0 or self.trainer.is_last_batch

        logits_sample, fmaps_sample = self.discriminator(x)
        logits_rec, fmaps_rec = self.discriminator(audio_hat)
        gen_loss = generator_loss(audio_hat, x, fmaps_rec, fmaps_sample, logits_rec)
        disc_loss = discriminator_loss(logits_sample, logits_rec)

        if self.train_discriminator:
            with optimizer_d.toggle_model(sync_grad=is_last_batch_to_accumulate):
                self.log("train/discriminator/total", disc_loss, prog_bar=True)
                loss = disc_loss / self.accumulate_grad_batches

                self.manual_backward(loss, retain_graph=True)

                if is_last_batch_to_accumulate:
                    self.clip_gradients(
                        optimizer_d, gradient_clip_val=1, gradient_clip_algorithm="norm"
                    )
                    optimizer_d.step()
                    optimizer_d.zero_grad()
                    scheduler_d.step()

        with optimizer_g.toggle_model(sync_grad=is_last_batch_to_accumulate):
            commit_loss = commit_loss.mean()

            loss = (
                sum(v for v in gen_loss.values()) + commit_loss
            ) / self.accumulate_grad_batches

            if self.use_balancer:
                self.balancer.backward(
                    gen_loss,
                    audio_hat,
                )
                self.manual_backward(commit_loss / self.accumulate_grad_batches)
            else:
                self.manual_backward(loss / self.accumulate_grad_batches)

            if is_last_batch_to_accumulate:
                self.clip_gradients(
                    optimizer_g, gradient_clip_val=1, gradient_clip_algorithm="norm"
                )
                optimizer_g.step()
                optimizer_g.zero_grad()
                scheduler_g.step()

        self.log("train/generator/total_loss", loss, prog_bar=True)
        self.log("train/generator/reconstruction_loss", gen_loss["reconstruction_loss"])
        self.log("train/generator/line_loss", gen_loss["line_loss"])
        self.log("train/generator/spectral_loss", gen_loss["spectral_loss"])
        self.log("train/generator/feature_loss", gen_loss["feature_loss"])
        self.log("train/generator/adversarial_loss", gen_loss["adversarial_loss"])
        self.log("train/commit_loss", commit_loss)
        self.log("train/discriminator/lr", scheduler_d.get_last_lr()[0])
        self.log("train/generator/lr", scheduler_g.get_last_lr()[0])

        batch_prd = self._compute_prd(x, audio_hat)
        self.train_prd(batch_prd)
        self.log("train/prd", self.train_prd, prog_bar=True)

        loss = loss / self.trainer.accumulate_grad_batches

    def test_step(self, batch, batch_idx):
        x = batch.data

        audio_hat, _ = self._common_step(x)

        x = rearrange(
            x,
            "batch channel length -> (batch channel) 1 length",
        )

        batch_prd = self._compute_prd(x, audio_hat)
        self.test_prd(batch_prd)
        self.log("test/prd", self.test_prd)

        return

    @staticmethod
    def _compute_prd(x: torch.Tensor, y: torch.Tensor):
        x = x.detach().clone()
        y = y.detach().clone()

        residual = x.detach() - y
        batch_prd = (
            torch.sqrt((residual**2).sum(-1) / torch.clamp((x**2).sum(-1), min=1e-6))
            * 100
        )

        return batch_prd

    def configure_optimizers(self):
        disc_params = [
            {"params": self.discriminator.parameters()},
        ]
        gen_params = [
            {"params": self.model.parameters()},
            {"params": self.quantizer.parameters()},
        ]

        optimizer_disc = FusedAdam(
            disc_params,
            lr=self.lr if isinstance(self.lr, float) else self.lr.get("d", 0),
            weight_decay=0.01,
        )
        optimizer_gen = FusedAdam(
            gen_params,
            lr=self.lr if isinstance(self.lr, float) else self.lr.get("g", 0),
            weight_decay=0.01,
        )
        scheduler_disc = get_cosine_schedule_with_warmup(
            optimizer_disc,
            num_warmup_steps=self.warmup // self.accumulate_grad_batches,
            num_training_steps=self.training_steps // self.accumulate_grad_batches,
        )
        scheduler_gen = get_cosine_schedule_with_warmup(
            optimizer_gen,
            num_warmup_steps=self.warmup // self.accumulate_grad_batches,
            num_training_steps=self.training_steps // self.accumulate_grad_batches,
        )

        return (
            [optimizer_disc, optimizer_gen],
            [
                {"scheduler": scheduler_disc, "interval": "step"},
                {"scheduler": scheduler_gen, "interval": "step"},
            ],
        )
