import glob
import logging
import os
import pathlib
import sys

import datasets
import numpy as np
import torch
import transformers

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils_nlp.arguments import get_args_from_config
from utils_shared.constants import DATASET_NAMES_GLUE, DATASET_NAMES_SUPERGLUE
from utils_shared.run_utils_shared import get_log_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


BATCH_SIZES = {
    "cb": 8,
    "rte": 8,
    "default": 32,
}


def prepare_args(config, cli_args):
    model_args, data_args, training_args = get_args_from_config(config=config)
    training_args.seed = cli_args.seed
    model_args.model_seed = cli_args.seed
    training_args.optimizer_name = cli_args.optimizer_name
    data_args.dataset_name = cli_args.dataset_name
    data_args.task_name = (
        "superglue" if data_args.dataset_name in DATASET_NAMES_SUPERGLUE else "glue"
    )
    training_args.training_mode = cli_args.training_mode
    training_args.per_device_train_batch_size = BATCH_SIZES.get(
        data_args.dataset_name, BATCH_SIZES["default"]
    )
    training_args.adam_epsilon = cli_args.adam_epsilon
    model_args.model_name_or_path = cli_args.model_name
    if cli_args.lr_scheduler_type not in ["default", "linear"]:
        if training_args.training_mode != "normal":
            raise ValueError(
                f"lr_scheduler_type {cli_args.lr_scheduler_type} is not supported for training_mode {training_args.training_mode}"
            )
        training_args.lr_scheduler_type = cli_args.lr_scheduler_type
    training_args.output_dir = get_log_dir(cli_args)
    if cli_args.num_train_epochs is not None:
        training_args.num_train_epochs = cli_args.num_train_epochs
    training_args.gradient_accumulation_steps = cli_args.accumulation_steps
    return model_args, data_args, training_args


def setup_library(training_args):
    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    datasets.logging.disable_progress_bar()
    training_args.disable_tqdm = True
    return training_args


def train(trainer):
    train_result = trainer.train()
    if train_result is None:
        return
    metrics = train_result.metrics
    if len(metrics) == 0:
        return
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def evaluate(trainer):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def predict(trainer, predict_dataset=None):
    logger.info("*** Predict ***")
    predictions = trainer.predict(predict_dataset)

    trainer.log_metrics("test", predictions.metrics)
    trainer.save_metrics("test", predictions.metrics)
    trainer.log(predictions.metrics)
    return


def get_my_last_checkpoint(directory):
    # Since the importing of transformers is slow, we do it here
    checkpoints = glob.glob(f"{directory}/checkpoint-*")
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    last_checkpoint = checkpoints[-1]
    return last_checkpoint


def check_output_dir(training_args):
    to_continue = True
    if training_args.do_train:
        output_dir_path = pathlib.Path(training_args.output_dir)
        if output_dir_path.exists() and any(output_dir_path.iterdir()):
            if not training_args.overwrite_output_dir:
                logger.info(
                    f" `file exists in ({training_args.output_dir}) and --overwrite_output_dir` is not True, so exit."
                )
                to_continue = False
            else:
                logger.info(
                    f" file exists in ({training_args.output_dir}), but `--overwrite_output_dir` is True, so continue."
                )
    return to_continue
