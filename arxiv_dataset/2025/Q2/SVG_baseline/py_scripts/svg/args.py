import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument("--dataset_name",
                        type=str,
                        default="landscape",
                        choices=["landscape", "greatesthits"],
                        help=(
                            "The name of the Dataset."
                        ),
    )
    parser.add_argument("--max_train_samples",
                        type=int,
                        default=None,
                        help=(
                            "For debugging purposes or quicker training, truncate the number of training examples to this "
                            "value if set."
                        ),
    )
    parser.add_argument("--output_dir",
                        type=str,
                        default="out_tmp",
                        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", 
                        type=int, 
                        default=None, 
                        help="A seed for reproducible training.")
    parser.add_argument("--resolution",
                        type=int,
                        default=256,
                        help=(
                            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                            " resolution"
                        ),
    )
    parser.add_argument("--center_crop",
                        default=False,
                        action="store_true",
                        help=(
                            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
                            " cropped. The images will be resized to the resolution first before cropping."
                        ),
    )
    parser.add_argument("--random_flip",
                        action="store_true",
                        help="whether to randomly flip images horizontally",
    )
    parser.add_argument("--num_frames_per_sample",
                        type=int, 
                        default=16, 
                        help="# of video frames per sample for training."
    )
    parser.add_argument("--duration_per_sample", 
                        type=int, 
                        default=4, 
                        help="# of video duration per sample for training."
    )
    parser.add_argument("--train_batch_size", 
                        type=int, 
                        default=16, 
                        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", 
                        type=int, 
                        default=100)
    parser.add_argument("--max_train_steps",
                        type=int,
                        default=None,
                        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--gradient_checkpointing",
                        action="store_true",
                        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--scale_lr",
                        action="store_true",
                        default=False,
                        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument("--lr_scheduler",
                        type=str,
                        default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
    )
    parser.add_argument("--lr_warmup_steps",
                        type=int, 
                        default=500, 
                        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--use_8bit_adam", 
                        action="store_true", 
                        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--allow_tf32",
                        action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
    )
    parser.add_argument("--use_ema", 
                        action="store_true", 
                        help="Whether to use EMA model.")
    parser.add_argument("--dataloader_num_workers",
                        type=int,
                        default=4,
                        help=(
                            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
                        ),
    )
    parser.add_argument("--adam_beta1", 
                        type=float, 
                        default=0.9, 
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", 
                        type=float, 
                        default=0.999, 
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", 
                        type=float, 
                        default=1e-2, 
                        help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", 
                        type=float, 
                        default=1e-08, 
                        help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", 
                        default=1.0, 
                        type=float, 
                        help="Max gradient norm.")
    parser.add_argument("--logging_dir",
                        type=str,
                        default="logs",
                        help=(
                            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
                        ),
    )
    parser.add_argument("--mixed_precision",
                        type=str,
                        default=None,
                        choices=["no", "fp16", "bf16"],
                        help=(
                            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
                        ),
    )
    parser.add_argument("--local_rank", 
                        type=int, 
                        default=-1, 
                        help="For distributed training: local_rank")
    parser.add_argument("--report_to",
                        type=str,
                        default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
    )
    parser.add_argument("--checkpointing_steps",
                        type=int,
                        default=10000,
                        help=(
                            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
                            " training using `--resume_from_checkpoint`."
                        ),
    )
    parser.add_argument("--checkpoints_total_limit",
                        type=int,
                        default=3,
                        help=(
                            "Max number of checkpoints to store. Will be ignored if set to 0 or negative values."
                        ),
    )
    parser.add_argument("--resume_from_checkpoint",
                        type=str,
                        default=None,
                        help=(
                            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
                            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
                        ),
    )
    parser.add_argument("--validation_epochs",
                        type=int,
                        default=5,
                        help="Run validation every X epochs.",
    )
    parser.add_argument("--num_validation_samples",
                        type=int,
                        default=5,
                        help="The number of samples used for validation.",
    )
    parser.add_argument("--tracker_project_name",
                        type=str,
                        default="svg",
                        help=(
                            "The `project_name` argument passed to Accelerator.init_trackers for"
                            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
                        ),
    )
    parser.add_argument("--drop_rate_cfg", 
                        type=float, 
                        default=0.1, 
                        help="The dropout rate for classifier-free guidance."
    )
    parser.add_argument("--how_to_drop_cond", 
                        type=str, 
                        default="zero", 
                        help="How to drop video conditioning. zero for inputing zeros to cross attention layers, none for inputing nothing."
    )
    parser.add_argument("--connector_in_type", 
                        type=str,
                        default="additive", 
                        help="how to feed connector features into the model. attn or additive"
    )
    parser.add_argument("--finetune_unet", 
                        action="store_true", 
                        help="Fix the pretrained unet through the training."
    )
    parser.add_argument("--fix_temporal", 
                        action="store_true", 
                        help="Fix temporal layers in the t2v unet."
    )
    parser.add_argument("--video_model", 
                        type=str,
                        default="animatediff", 
                        help="The pretrained text-to-video model used for training. animatediff / modelscope / zeroscope"
    )
    parser.add_argument("--use_xt_for_connectors", 
                        action="store_true", 
                        help="Use x0 prediction for the input of connectors."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    args.use_x0_pred_for_connectors = not args.use_xt_for_connectors

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args

