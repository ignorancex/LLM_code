import os
import shutil
import sys

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from wyckoff_generation.common.constants import (
    BEST_PARAMS_FILE,
    PARAMS_FILE,
    WANDB_PROJECT,
)
from wyckoff_generation.common.registry import registry


class BaseLogger:
    def __init__(self, config):
        print(
            f"Setting up logger, using {self.__class__.__qualname__}", file=sys.stdout
        )
        if not os.path.isdir("checkpoints"):
            os.mkdir("checkpoints")
        if not os.path.isdir("logs"):
            os.mkdir("logs")

    def log(self, data_to_log, step):
        raise NotImplementedError

    def save_checkpoint(self, dict_to_save, best, epoch=None):
        raise NotImplementedError

    def update_summary(self, log):
        raise NotImplementedError


@registry.register_logger("none")
@registry.register_logger("noop")
class NoOpLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)

    def log(self, data_to_log, step):
        return

    def save_checkpoint(self, dict_to_save, best, epoch=None):
        return

    def update_summary(self, log):
        return


def create_checkpoint_directory(run_name, load_path: str):

    if load_path:
        split_path = load_path.split("/")
        assert (
            "checkpoints" in split_path
        ), "Ensure that load parameters is in checkpoints directory."
        # # Use if we want to old directory
        # # Get up until directory only
        # checkpoint_dir = split_path[0:-1]
        # TODO: If we want this style, rewrite and put the assert below here and check for "if not load_path: ..."
        checkpoint_dir = os.path.join("checkpoints", run_name)
        os.mkdir(checkpoint_dir)
    else:
        checkpoint_dir = os.path.join("checkpoints", run_name)
        assert not os.path.isdir(checkpoint_dir), "Checkpoint directory already exists"
        os.mkdir(checkpoint_dir)
    return checkpoint_dir


def create_log_directory(run_name, load_path: str):
    if load_path:
        split_path = load_path.split("/")
        assert (
            "checkpoints" in split_path
        ), "Ensure that load parameters is in checkpoints directory."
        # # Use if we want to old directory
        for idx, element in enumerate(split_path):
            if element == "checkpoints":
                # Get next element which should be checkpoint dir.
                # Get up until last directory only
                prev_log_dir = os.path.join("logs", *split_path[idx + 1 : -1])
        # TODO: If we want this style, rewrite and put the assert below here and check for "if not load_path: ..."
        log_dir = os.path.join("logs", run_name)
        # Create new logdir
        os.mkdir(log_dir)

        # Copy previous log files to new dir.
        for log_file in os.listdir(prev_log_dir):
            shutil.copy(os.path.join(prev_log_dir, log_file), log_dir)

    else:
        log_dir = os.path.join("logs", run_name)
        assert not os.path.isdir(log_dir), "Log directory already exists"
        os.mkdir(log_dir)
    return log_dir


@registry.register_logger("model_only")
class ModelOnlyLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)

        # create directory in checkpoints/
        self.checkpoint_dir = create_checkpoint_directory(
            config["run_name"], config["load"]
        )

    def update_summary(self, log):
        pass

    def log(self, data_to_log, step):
        pass

    def save_checkpoint(self, dict_to_save, best, epoch=None):
        if best:
            name = BEST_PARAMS_FILE
        elif epoch is not None:
            name, ext = os.path.splitext(PARAMS_FILE)
            name = f"{name}_{epoch}{ext}"
        else:
            name = PARAMS_FILE
        torch.save(dict_to_save, os.path.join(self.checkpoint_dir, name))


@registry.register_logger("local_only")
class LocalOnlyLogger(ModelOnlyLogger):
    def __init__(self, config):
        super().__init__(config)

        # create directory in logs/ where all logging is saved
        self.log_dir = create_log_directory(config["run_name"], config["load"])

        # create log file
        self.log_file = os.path.join(self.log_dir, "full_log.out")

    def update_summary(self, log):
        pass

    def log(self, data_to_log, step):
        full_dict = {"Step": step}
        full_dict.update(data_to_log)
        with open(self.log_file, "a") as log_file:
            log_file.write(f"{full_dict}\n")


@registry.register_logger("wandb")
class WandBLogger(ModelOnlyLogger):
    def __init__(self, config):
        super().__init__(config)

        # create directory in logs/ where all logging is saved
        self.log_dir = create_log_directory(config["run_name"], config["load"])

        wandb.init(
            project=WANDB_PROJECT,
            config=config,
            name=config["run_name"],
            dir=self.log_dir,
        )

    def log(self, data_to_log, step):
        wandb.log(data_to_log, step=step)

    def update_summary(self, log):
        for key, value in log.items():
            wandb.run.summary[key] = value


@registry.register_logger("tensorboard")
class TensorBoardLogger(ModelOnlyLogger):

    """
    PyTorch compatible TensorBoard logger class.
    When training is finished run: "$ tensorboard --logdir=logs/<trainingsession_dir>" to plot the data.
    """

    def __init__(self, config):
        super().__init__(config)

        # create directory in logs/ where all logging is saved
        self.log_dir = create_log_directory(config["run_name"], config["load"])

        # PyTorch tensorboard logger
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def __del__(self):
        print(
            "Tensorboard logger ensuring pending events are written to disk...",
            file=sys.stdout,
        )
        self.writer.flush()

    def log(self, data_to_log: dict, step: int):
        for tag, value in data_to_log.items():
            self.writer.add_scalar(tag, value, step)

    def update_summary(self, log):
        pass
