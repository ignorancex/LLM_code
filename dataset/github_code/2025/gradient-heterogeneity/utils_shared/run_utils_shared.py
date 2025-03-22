import argparse
import logging
import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return worker_init_fn


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="nlp")
    parser.add_argument(
        "--arguments", type=str, default="conf/roberta-base/arguments.yaml"
    )
    parser.add_argument("--optimizer_name", type=str, default="adam")
    parser.add_argument("--dataset_name", type=str, default="cb")
    parser.add_argument("--training_mode", type=str, default="normal")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--hyperparameter_config_dir",
        type=str,
        default="conf",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--script_type", type=str, default="evaluation")
    parser.add_argument("--tuning_metric", type=str, default="train_loss")
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--lr_scheduler_type", type=str, default="default")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    return parser.parse_args()


def get_log_dir(cli_args):
    sub_dir_name = cli_args.training_mode
    if cli_args.lr_scheduler_type not in ["linear", "default"]:
        sub_dir_name = cli_args.lr_scheduler_type
    log_dir = os.path.join(
        "results",
        cli_args.script_type,
        cli_args.domain,
        cli_args.dataset_name,
        cli_args.model_name,
        sub_dir_name,
        cli_args.optimizer_name,
        f"seed_{cli_args.seed}",
    )
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_hyperparameter_config_path(cli_args):
    training_mode_or_lr_scheduler_type = (
        cli_args.training_mode
        if cli_args.lr_scheduler_type == "default"
        else cli_args.lr_scheduler_type
    )
    config_path = os.path.join(
        cli_args.hyperparameter_config_dir,
        cli_args.model_name,
        f"{training_mode_or_lr_scheduler_type}.yaml",
    )
    return config_path
