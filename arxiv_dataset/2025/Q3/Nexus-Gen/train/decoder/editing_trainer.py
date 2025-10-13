import argparse
import yaml
import os
import torch
import lightning as pl
from transformers import get_constant_schedule_with_warmup, AutoConfig
from modeling.decoder.modules import ImageEmbeddingMerger
from diffsynth import ModelManager
from diffsynth.models.utils import load_state_dict
from train.decoder.embed_image_dataset import EditingDecoderDataset
from modeling.decoder.pipelines import NexusGenEditingPipeline


class EditingDecoder(pl.LightningModule):
    def __init__(
        self,
        torch_dtype=torch.float16, pretrained_weights=[], preset_lora_path=None,
        learning_rate=1e-4, use_gradient_checkpointing=True, state_dict_converter=None,
        quantize = None, qwenvl_path=None, out_channel=4096, lr_warmup_steps=500, load_from=None, num_layers=1,
    ):
        super().__init__()
        # Set parameters
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.state_dict_converter = state_dict_converter
        self.lora_alpha = None
        self.lr_warmup_steps = lr_warmup_steps
        # Load models
        model_manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        if quantize is None:
            model_manager.load_models(pretrained_weights)
        else:
            model_manager.load_models(pretrained_weights[1:])
            model_manager.load_model(pretrained_weights[0], torch_dtype=quantize)
        if preset_lora_path is not None:
            model_manager.load_lora(preset_lora_path)

        self.pipe = NexusGenEditingPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        model_config = AutoConfig.from_pretrained(qwenvl_path)
        self.embedding_merger = ImageEmbeddingMerger(model_config, num_layers=num_layers, out_channel=out_channel)
        attention_state_dict = load_state_dict(os.path.join(qwenvl_path, 'model-00001-of-00005.safetensors'))
        attention_state_dict = {k.replace('model.layers.', ''): v for k, v in attention_state_dict.items() if k.startswith('model.layers')}
        self.embedding_merger.layers.load_state_dict(attention_state_dict, strict=False)

        if load_from is not None:
            state_dict = torch.load(load_from, weights_only=True, map_location="cpu")
            dit_state_dict = {key.replace("pipe.dit.", ""): value for key, value in state_dict.items() if key.startswith('pipe.dit.')}
            embedding_merger_state_dict = {key.replace("embedding_merger.", ""): value for key, value in state_dict.items() if key.startswith('embedding_merger.')}
            self.embedding_merger.load_state_dict(embedding_merger_state_dict)
            self.pipe.denoising_model().load_state_dict(dit_state_dict)

        self.freeze_parameters()


    def load_models(self):
        # This function is implemented in other modules
        self.pipe = None


    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().requires_grad_(True)
        self.pipe.denoising_model().train()
        self.embedding_merger.requires_grad_(True)
        self.embedding_merger.train()


    def training_step(self, batch, batch_idx):
        embed, image, ref_embed, embeds_grid, ref_embeds_grid = batch["embed"], batch["image"], batch["ref_embed"], batch["embeds_grid"], batch["ref_embeds_grid"]
        # Prepare input parameters
        self.pipe.device = self.device
        prompt_emb = self.pipe.encode_prompt("", positive=True, clip_only=True)
        visual_emb = self.embedding_merger(embed, embeds_grid, ref_embed, ref_embeds_grid)
        prompt_emb['prompt_emb'] = visual_emb

        if "latents" in batch:
            latents = batch["latents"].to(dtype=self.pipe.torch_dtype, device=self.device)
        else:
            latents = self.pipe.vae_encoder(image.to(dtype=self.pipe.torch_dtype, device=self.device))

        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        extra_input = self.pipe.prepare_extra_input(latents, guidance=3.5)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # prepare embed_ids
        batch_size, height, width = image.shape[0], embeds_grid[0][1], embeds_grid[0][2]
        embed_ids = torch.zeros(height // 2, width // 2, 3)
        scale_factor_height, scale_factor_width = noisy_latents.shape[-2] / height, noisy_latents.shape[-1] / width
        embed_ids[..., 1] = embed_ids[..., 1] + torch.arange(height // 2)[:, None] * scale_factor_height.item()
        embed_ids[..., 2] = embed_ids[..., 2] + torch.arange(width // 2)[None, :] * scale_factor_width.item()
        embed_ids = embed_ids[None, :].repeat(batch_size, 1, 1, 1).reshape(batch_size, height // 2 * width // 2, 3)
        embed_text_ids = embed_ids.to(device=latents.device, dtype=latents.dtype)

        batch_size, height, width = image.shape[0], ref_embeds_grid[0][1], ref_embeds_grid[0][2]
        ref_embed_ids = torch.zeros(height // 2, width // 2, 3)
        scale_factor_height, scale_factor_width = noisy_latents.shape[-2] / height, noisy_latents.shape[-1] / width
        ref_embed_ids[..., 0] = ref_embed_ids[..., 0] + 1.0
        ref_embed_ids[..., 1] = ref_embed_ids[..., 1] + torch.arange(height // 2)[:, None] * scale_factor_height.item()
        ref_embed_ids[..., 2] = ref_embed_ids[..., 2] + torch.arange(width // 2)[None, :] * scale_factor_width.item()
        ref_embed_ids = ref_embed_ids[None, :].repeat(batch_size, 1, 1, 1).reshape(batch_size, height // 2 * width // 2, 3)
        ref_embed_text_ids = ref_embed_ids.to(device=latents.device, dtype=latents.dtype)

        prompt_emb['text_ids'] = torch.cat([embed_text_ids, ref_embed_text_ids], dim=1)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train_learning_rate", lr, prog_bar=True, on_step=True, on_epoch=False, rank_zero_only=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, rank_zero_only=True)
        return loss


    def configure_optimizers(self):
        import itertools
        trainable_modules = itertools.chain(
            self.embedding_merger.parameters(),
            self.pipe.denoising_model().parameters()
        )
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
    
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.lr_warmup_steps
        print('total_steps:', total_steps)

        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        dit_state_dict = self.pipe.denoising_model().state_dict()
        dit_state_dict = {f"pipe.dit.{key}": value for key, value in dit_state_dict.items()}
        checkpoint.update(dit_state_dict)

        merger_state_dict = self.embedding_merger.state_dict()
        merger_state_dict = {f"embedding_merger.{key}": value for key, value in merger_state_dict.items()}
        checkpoint.update(merger_state_dict)


def launch_training_task(model, args):
    print(args)
    # dataset and data loader
    dataset = EditingDecoderDataset(
        args.dataset_path,
        steps_per_epoch=args.steps_per_epoch,
        height=args.height,
        width=args.width,
        center_crop=args.center_crop,
        random_flip=args.random_flip
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers
    )
    # train
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision=args.precision,
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=None,
        log_every_n_steps=5,
    )
    trainer.fit(model=model, train_dataloaders=train_loader)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=False,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width.",
    )
    parser.add_argument(
        "--center_crop",
        default=True,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="Whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16", "16-mixed", "bf16"],
        help="Training precision",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=1,
        help="lr_warmup_steps.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/train/flux/controlnet.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--pretrained_text_encoder_path",
        type=str,
        default='models/FLUX/FLUX.1-dev/text_encoder/model.safetensors',
        help="Path to pretrained text encoder model. For example, `models/FLUX/FLUX.1-dev/text_encoder/model.safetensors`.",
    )
    parser.add_argument(
        "--qwenvl_path",
        type=str,
        default='',
        help="Path to pretrained qwen_vl. For example, ``.",
    )
    parser.add_argument(
        "--pretrained_dit_path",
        type=str,
        default='models/FLUX/FLUX.1-dev/flux1-dev.safetensors',
        help="Path to pretrained dit model. For example, `models/FLUX/FLUX.1-dev/flux1-dev.safetensors`.",
    )
    parser.add_argument(
        "--pretrained_vae_path",
        type=str,
        default='models/FLUX/FLUX.1-dev/ae.safetensors',
        help="Path to pretrained vae model. For example, `models/FLUX/FLUX.1-dev/ae.safetensors`.",
    )
    parser.add_argument(
        "--load_from",
        type=str,
        default=None,
        help="Path to pretrained dit and adapter.",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Update args with YAML configuration
    for key, value in config.items():
        setattr(args, key, value)
    return args


if __name__ == '__main__':
    args = parse_args()
    load_from = None if args.load_from is None or args.load_from.lower() == 'none' else args.load_from
    model = EditingDecoder(
        torch_dtype={"32": torch.float32, "bf16": torch.bfloat16}.get(args.precision, torch.float16),
        pretrained_weights=[args.pretrained_dit_path, args.pretrained_text_encoder_path, args.pretrained_vae_path],
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        lr_warmup_steps=args.lr_warmup_steps,
        qwenvl_path=args.qwenvl_path,
        load_from=load_from,
    )
    launch_training_task(model, args)
