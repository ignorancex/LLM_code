import os
import glob
import torch
import wandb
from loguru import logger
from train.distributed import init_distributed_device, world_info_from_env
from train.data import get_websight_dataset
from train.train_utils import get_cast_dtype, get_mp_policy_dtype, AverageMeter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

def setup_distributed_training(args):
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    return device_id

def setup_model_and_optimizer(args, model, device_id):
    if args.fsdp:
        ddp_model = setup_fsdp_model(model, args, device_id)
    else:
        model = model.to(device_id)
        ddp_model = DDP(model, device_ids=[device_id])

    optimizer = setup_optimizer(ddp_model, args)
    lr_scheduler = setup_lr_scheduler(optimizer, args)

    return ddp_model, optimizer, lr_scheduler

def setup_fsdp_model(model, args, device_id):
    # Implementation of FSDP setup
    # This is a placeholder and should be implemented based on your FSDP requirements
    pass

def setup_optimizer(ddp_model, args):
    params_to_optimize = ddp_model.named_parameters()
    params_to_optimize = list(
        filter(
            lambda x: x[1].requires_grad
            and not getattr(x[1], "exclude_from_optimizer", False),
            params_to_optimize,
        )
    )
    if not args.fsdp or args.fsdp_use_orig_params:
        optimizer = torch.optim.AdamW(
            get_grouped_params(params_to_optimize, args),
            lr=args.learning_rate
        )
    else:
        optimizer = torch.optim.AdamW(
            (p for _, p in params_to_optimize),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    return optimizer

def get_grouped_params(params_to_optimize, args):
    params_with_wd, params_without_wd = [], []
    for n, p in params_to_optimize:
        if "gated_cross_attn" in n:
            params_with_wd.append(p)
        else:
            params_without_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": args.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def setup_lr_scheduler(optimizer, args):
    if args.lr_scheduler == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_training_steps,
        )
    else:
        return get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

def setup_data_and_training_params(args, image_processor, clip_model, tokenizer):
    websight_loader, steps_per_epoch = get_websight_dataset(
        args.batch_size_websight, 
        image_processor, 
        clip_model, 
        tokenizer,
        args.vision_encoder_path,
        num_samples=args.train_num_samples_websight
    )
    total_training_steps = args.num_epochs * steps_per_epoch
    args.total_training_steps = total_training_steps
    args.steps_per_epoch = steps_per_epoch
    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")
        print(f"Steps/batches per epoch: {steps_per_epoch}")
    return websight_loader, total_training_steps, steps_per_epoch

def setup_logging(args):
    print(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
        logger.info("Init wandb")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
        logger.info("Completely initialize wandb")

def load_checkpoint(args, model, optimizer, lr_scheduler):
    if os.path.exists(f"./model/{args.run_name}") and args.resume_from_checkpoint is None:
        checkpoint_list = glob.glob(f"./model/{args.run_name}/checkpoint_*.pt")
        if len(checkpoint_list) > 0:
            args.resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}.")

    if args.resume_from_checkpoint:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        return start_epoch
    return 0
