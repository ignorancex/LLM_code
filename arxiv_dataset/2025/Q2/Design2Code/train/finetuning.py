""" Main training script """

import argparse
import random
from loguru import logger
import numpy as np
import torch
import wandb
from train.train_utils import (
    train_one_epoch,
    save_checkpoint
)
from train.utils import (
    setup_distributed_training,
    setup_model_and_optimizer,
    setup_data_and_training_params,
    setup_logging,
    load_checkpoint,
)
from src.factory import create_model_and_transforms

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    # Add all the existing arguments here
    # Model configuration args
    parser.add_argument("--enable_graph_input", action="store_true")
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument("--tokenizer_path", default="facebook/opt-30b", type=str)
    parser.add_argument("--cross_attn_every_n_layers", type=int, default=1)

    # Training args
    parser.add_argument("--run_name", type=str, default="openflamingo3B")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--delete_previous_checkpoint", action="store_true")
    parser.add_argument("--batch_size_websight", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--loss_multiplier_websight", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--precision", choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"], default="fp32")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--freeze_lm_embeddings", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_ckpt_steps", type=int, default=10000)

    # Data args
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train_num_samples_websight", type=int, default=10000)
    parser.add_argument("--dataset_resampled", action="store_true")

    # Distributed training args
    parser.add_argument("--dist-url", default="env://", type=str)
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--horovod", default=False, action="store_true")
    parser.add_argument("--no-set-device-rank", default=False, action="store_true")
    parser.add_argument("--fsdp", default=False, action="store_true")
    parser.add_argument("--fsdp_use_orig_params", default=False, action="store_true")
    parser.add_argument("--fsdp_sharding_strategy", default="full", type=str, choices=["full", "hybrid"])

    # Wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--save_checkpoints_to_wandb", default=False, action="store_true")

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set up distributed training
    device_id = setup_distributed_training(args)
    random_seed(args.seed, args.rank)
    
    # Initialize model, optimizer, and scheduler
    model, image_processor, clip_model, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
        enable_graph_input=args.enable_graph_input,
    )
    ddp_model, optimizer, lr_scheduler = setup_model_and_optimizer(args, model, device_id)
    
    # Initialize data loaders and training parameters
    websight_loader, total_training_steps, steps_per_epoch = setup_data_and_training_params(
        args, image_processor, clip_model, tokenizer
    )
    
    # Initialize logging
    setup_logging(args)

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_from_checkpoint:
        start_epoch = load_checkpoint(args, ddp_model, optimizer, lr_scheduler)

    # Start training
    ddp_model.train()
    if args.rank == 0:
        logger.info("Start training...")
    for epoch in range(start_epoch, args.num_epochs):
        train_one_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device_id=device_id,
            wandb=wandb,
            websight_loader=websight_loader,
            enable_graph_input=args.enable_graph_input,
        )
        save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, (epoch+1)*steps_per_epoch-1, args)

    # Save final checkpoint
    save_checkpoint(ddp_model, optimizer, lr_scheduler, args.num_epochs-1, total_training_steps-1, args)

if __name__ == "__main__":
    main()
