import torch
from transformers import TrainingArguments, Trainer
import numpy as np
import argparse
import os
from datetime import datetime
import json
import wandb
import torch.distributed as dist
from accelerate import Accelerator

from utils.data_utils import (
    load_weighted_it,
    load_and_preprocess_it,
    WeightedDataCollatorForSupervisedDataset,
    DataCollatorForSupervisedDataset,
)
from models import create_model_tokenizer_it, create_peft_model_it, IGNORE_INDEX
from utils.misc import count_parameters
from utils.trainer_utils import WeightedLossTrainer, SFATrainer
from utils.parsing_utils import str_to_bool


def init_distributed():
    if dist.is_initialized():
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device(rank)
        return rank, world_size
    else:
        print("Distributed environment variables not set, running in non-distributed mode")
        return 0, 1


def freeze_parameters(model):
    """Freeze all parameters except the head/output layer."""
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False


def freeze_non_head_parameters(model):
    """Freeze all parameters except the head/output layer."""
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the head/output layer parameters
    for param in model.lm_head.parameters():
        param.requires_grad = True


def create_run_directory(args):
    """Create a directory structure for the current training run."""
    base_dir = args.base_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.split("/")[-1]
    run_name = f"{model_name}__r{args.lora_r}__lr{args.lr}__ft_{args.finetune_type}_reg_{args.weight_regularization}__train_{args.dataset_split.replace('[:','').replace(']','')}"
    run_dir = os.path.join(base_dir, model_name, f"{timestamp}_{run_name}")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    config_dict = vars(args)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    return run_dir


def finetune():
    local_rank, world_size = init_distributed()
    accelerator = Accelerator()

    if local_rank == 0:
        run_dir = create_run_directory(args)
    else:
        run_dir = None

    if dist.is_initialized():
        # Broadcast the run directory path from rank 0 to all other ranks
        if local_rank == 0:
            run_dir_path = run_dir
        else:
            run_dir_path = None

        run_dir_path = [run_dir_path]  # Wrap in list for broadcast_object_list
        dist.broadcast_object_list(run_dir_path, src=0)
        run_dir = run_dir_path[0]

    if not args.no_wandb and local_rank == 0:
        wandb_run_name = os.path.basename(run_dir)
        wandb_run = wandb.init(project=args.project_name, config=args, dir=os.path.join(run_dir, "logs"))
        with open(os.path.join(run_dir, "wandb_run_id.txt"), "w") as f:
            f.write(wandb_run.id)

    training_args = TrainingArguments(
        output_dir=os.path.join(run_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        seed=args.seed,
        report_to="wandb" if not args.no_wandb else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy="no",
        # optim="adamw_torch",
        bf16=True,
        bf16_full_eval=True,
        logging_steps=1,
        logging_first_step=True,
        logging_dir=os.path.join(run_dir, "logs"),
        remove_unused_columns=False if args.reweight_type != "none" else True,
    )

    model, tokenizer = create_model_tokenizer_it(args)

    # Data handling
    if args.reweight_type != "none":
        train_dataset = load_weighted_it(args=args)
        data_collator = WeightedDataCollatorForSupervisedDataset(tokenizer=tokenizer, loss_type=args.reweight_type)
        data_module = dict(train_dataset=train_dataset, data_collator=data_collator)
    else:
        train_dataset = load_and_preprocess_it(tokenizer=tokenizer, args=args)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        data_module = dict(train_dataset=train_dataset, data_collator=data_collator)

    # Create peft model
    if args.finetune_type == "none":
        freeze_parameters(model)
        model = model.to(args.device)
    elif args.finetune_type == "linear-probe":
        freeze_non_head_parameters(model)
        model = model.to(args.device)
    elif args.finetune_type == "full":
        model = model.to(args.device)
    elif args.finetune_type == "lora":
        model, lora_config = create_peft_model_it(model, args)
    elif args.finetune_type == "sfa":
        model = model.to(args.device)
    else:
        raise RuntimeError(f"Unknown finetuning type {args.finetune_type}")

    if not args.no_wandb and local_rank == 0:
        param_counts = count_parameters(model, verbose=False)
        wandb.log(
            {
                "total_params": param_counts["total_trainable_params"],
                "classifier_params": param_counts["classifier_params"],
                "non_classifier_params": param_counts["non_classifier_params"],
            }
        )

    if local_rank == 0:
        training_args_path = os.path.join(run_dir, "training_args.json")
        with open(training_args_path, "w") as f:
            json.dump(training_args.to_dict(), f, indent=4)
        tokenizer.save_pretrained(os.path.join(run_dir, "tokenizer"))

    if args.reweight_type == "none" and args.weight_regularization == "none":
        if args.finetune_type == "sfa":
            model_copy = model.__class__(model.config)
            model_copy.load_state_dict(model.state_dict())
            model_copy = model_copy.to(args.device)
            model_state = model_copy.state_dict()
            trainer = SFATrainer(
                model=model,
                args=training_args,
                model_state=model_state,
                averaging_freq=args.sfa_update_freq,
                beta=args.sfa_beta,
                **data_module,
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                **data_module,
            )
    else:  # Use sequence or token weighted loss trainer or regularized loss trainer
        if args.weight_regularization != "none":
            model_copy = model.__class__(model.config)
            model_copy.load_state_dict(model.state_dict())
            model_copy = model_copy.to(args.device)
            trainer = WeightedLossTrainer(
                model=model,
                args=training_args,
                loss_type=args.reweight_type,
                ignore_index=IGNORE_INDEX,
                base_model=model_copy,
                weight_regularization=args.weight_regularization,
                reg_lambda=args.weight_regularization_lambda,
                **data_module,
            )
        else:
            trainer = WeightedLossTrainer(
                model=model, args=training_args, loss_type=args.reweight_type, ignore_index=IGNORE_INDEX, **data_module
            )

    model.config.use_cache = False
    trainer.train()

    if world_size > 1:
        dist.barrier()

    final_model_path = os.path.join(run_dir, "final_model")
    if local_rank == 0:
        trainer.save_state()

    if world_size > 1:
        dist.barrier()

    # If using FSDP we need to unwrap the model before saving
    unwrapped_model = accelerator.unwrap_model(model=model)
    unwrapped_model.save_pretrained(
        final_model_path,
        is_main_process=local_rank == 0,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )

    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer for arithmetic reasoning tasks")

    # Dataset arguments
    parser.add_argument(
        "--base-dir", type=str, default="experiments/instruction_tuning", help="The directory to save the run"
    )
    parser.add_argument("--data_path", type=str, default="meta-math/MetaMathQA", help="Path to the training data")
    parser.add_argument(
        "--dataset_split", type=str, default="train", help="Dataset split to use can also specify `[:500]`"
    )
    parser.add_argument(
        "--dataset_field", type=str, nargs="+", default=["query", "response"], help="Fields of dataset input and output"
    )
    parser.add_argument(
        "--reweight-type",
        type=str,
        default="none",
        choices=["none", "sequence", "token"],
        help="How to reweight samples [sequence/token]-wise",
    )
    parser.add_argument(
        "--weight-regularization",
        type=str,
        default="none",
        choices=["none", "l1", "l2"],
        help="Weight regularization parameter",
    )
    parser.add_argument(
        "--weight-regularization-lambda", type=float, default=0.0, help="Weight regularization strength"
    )

    # Model and Finetuning arguments
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Model name")
    parser.add_argument(
        "--finetune-type", type=str, choices=["none", "full", "lora", "linear-probe", "sfa"], help="Select finetune type"
    )

    # LoRA specific arguments
    parser.add_argument("--lora_r", type=int, default=96, help="LoRA R value")
    parser.add_argument("--lora_alpha", type=int, default=96, help="LoRA alpha value")
    parser.add_argument("--lora_dropout", type=float, default=0, help="LoRA dropout value")

    #SFA specific arguments
    parser.add_argument("--sfa-update-freq", type=float, default=0.1, help="The update freq to take a convex combination")
    parser.add_argument("--sfa-beta", type=float, default=0.5, help="The beta used to take a convex combination")

    # General training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=16, help="Number of gradient accumulation steps"
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--scheduler", type=str, default="cosine", help="Learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.00, help="Weight Decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--gradient-checkpointing", type=str, default="false", choices=["true", "false"], help="Use gradient checkpointing in training")

    # Logging arguments
    parser.add_argument("--no-wandb", type=str, choices=["true", "false"], help="Turn of wandb logging")
    parser.add_argument(
        "--project-name", type=str, default="project-name", help="The name of the wandb project to log to"
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="The name of the current run to be logged in the project"
    )

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.no_wandb = str_to_bool(args.no_wandb)
    args.gradient_checkpointing = str_to_bool(args.gradient_checkpointing)
    print(args.no_wandb)

    args.lora_alpha = args.lora_r

    run_dir = finetune()
