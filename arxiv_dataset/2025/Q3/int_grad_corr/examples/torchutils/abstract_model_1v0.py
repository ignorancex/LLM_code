"""
Abstract model class easing checkpoint management and enforcing determinism.
"""

import os

import numpy as np
import torch


# Utilities


def get_checkpoint_path(
    saves_path, epoch, checkpoint_n_digits=4, training_checkpoint=False
):
    """
    Return checkpoint path.
    """
    checkpoint_name = "checkpoint"
    if training_checkpoint:
        checkpoint_name += "_training"
    checkpoint_name = f"{checkpoint_name}-{epoch:0{checkpoint_n_digits}d}"
    return os.path.join(saves_path, checkpoint_name)


def get_checkpoint_data(
    saves_path,
    checkpoint,
    training,
    current_epoch,
    device,
    checkpoint_n_digits=4,
):
    """
    Return the best, the last, or a specific checkpoint (if available).
    """
    if checkpoint is None:
        return None
    assert checkpoint in ("last", "best") or isinstance(
        checkpoint, int
    ), "Checkpoint must be 'last', 'best', or an integer."
    checkpoints = []
    for f in sorted(os.listdir(saves_path)):
        if os.path.isfile(os.path.join(saves_path, f)):
            f_split = f.split("-")
            if f_split[0] == "checkpoint":
                checkpoints.append(int(f_split[-1]))
    if training and not checkpoints:
        return None
    if not checkpoints:
        raise ValueError("No checkpoint to restore.")
    if checkpoint == "last":
        checkpoint = checkpoints[-1]
    elif checkpoint == "best":
        checkpoint_data = torch.load(
            get_checkpoint_path(
                saves_path, checkpoints[-1], checkpoint_n_digits
            ),
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        checkpoint = checkpoint_data["best_epoch"]
    if checkpoint != current_epoch:
        if checkpoint not in checkpoints:
            raise ValueError(
                f"Checkpoint {checkpoint:0{checkpoint_n_digits}d} "
                "does not exist."
            )
        checkpoint_data = torch.load(
            get_checkpoint_path(saves_path, checkpoint, checkpoint_n_digits),
            map_location=device,
            weights_only=True,
        )
        return checkpoint_data
    return None


def freeze_network(func):
    """
    This wrapper freezes/unfreezes AbstractModel's main module parameters.
    """

    def wrapper(*args, **kwargs):
        for param in args[0].network.parameters():
            param.requires_grad_(False)
        res = func(*args, **kwargs)
        for param in args[0].network.parameters():
            param.requires_grad_(True)
        return res

    return wrapper


class _SeedGenerator:
    def __init__(self, seed=None, initial_epoch=0, n_digits=9):
        self.rng = np.random.default_rng(seed)
        self.high = 10**n_digits
        for _ in range(initial_epoch):
            self.get_seed()

    def get_seed(self):
        return int(self.rng.integers(self.high, size=1)[0])


# Abstract model


class AbstractModel:
    """
    Ease checkpoint management and enforce determinism.

    This is an abstract class that must be inherited by sub-classes.
    The sub-classes are expected to redefine the following attributes:
        - project_path
        - network (main torch.nn.Module)
    And optionally:
        - seed
        - dtype
        - maximum_saves
        - checkpoint_n_digits
        - optimizer
        - scheduler
    """

    # Default parameters : to be defined #######################################
    project_path = None
    # Default parameters : optional ############################################
    seed = None
    dtype = torch.float32
    maximum_saves = 1
    checkpoint_n_digits = 4

    def __init__(
        self,
        model_name,
        device=None,
        parameters=None,
        model_name_suffix="",
        enforce_determinism=True,
    ):
        # Enforce determinism
        if enforce_determinism:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if enforce_determinism and (self.seed is None):
            self.seed = 100
        # Update parameters
        if parameters is not None:
            for key, value in parameters.items():
                setattr(self, key, value)
        # Init logs, saves, results folders
        self.model_name = model_name
        self.model_name_suffix = model_name_suffix
        self.logs_path = None
        self.saves_path = None
        self.results_path = None
        self._init_paths()
        # Set state variables
        self.current_step = -1
        self.current_epoch = -1
        self.best_epoch = -1
        self.best_val_acc = 1e10
        # Set the seed of the default PyTorch random number generator used for
        # weights initialization
        if self.seed is not None:
            torch.manual_seed(self.seed)
        # Set PyTorch default dtype
        torch.set_default_dtype(self.dtype)
        # Set device
        self.device = torch.device("cpu")
        if isinstance(device, int):
            device = f"cuda:{device}"
        if isinstance(device, str):
            if (
                len(device) >= 4
                and device[:4] == "cuda"
                and (torch.cuda.is_available())
            ):
                self.device = torch.device(device)
                torch.cuda.device(self.device)
            elif device == "mps" and torch.backends.mps.is_available():
                self.device = torch.device(device)
            elif device == "cpu":
                self.device = torch.device(device)
            else:
                raise ValueError(f"Device {device} is not available.")
        # Network : to be defined ##############################################
        self.network = None
        # Optimizer & scheduler : optional #####################################
        self.optimizer, self.scheduler = None, None

    def _init_paths(self):
        logs_path = os.path.join(self.project_path, "logs")
        if not os.path.isdir(logs_path):
            os.mkdir(logs_path)
        saves_path = os.path.join(self.project_path, "saves")
        if not os.path.isdir(saves_path):
            os.mkdir(saves_path)
        results_path = os.path.join(self.project_path, "results")
        if not os.path.isdir(results_path):
            os.mkdir(results_path)
        self.logs_path = os.path.join(
            logs_path, self.model_name + self.model_name_suffix
        )
        if not os.path.isdir(self.logs_path):
            os.mkdir(self.logs_path)
        self.saves_path = os.path.join(
            saves_path, self.model_name + self.model_name_suffix
        )
        if not os.path.isdir(self.saves_path):
            os.mkdir(self.saves_path)
        self.results_path = os.path.join(
            results_path, self.model_name + self.model_name_suffix
        )
        if not os.path.isdir(self.results_path):
            os.mkdir(self.results_path)
        return self

    def checkpoint_path(self, epoch, training_checkpoint=False):
        """
        Return checkpoint path.
        """

        return get_checkpoint_path(
            self.saves_path,
            epoch,
            self.checkpoint_n_digits,
            training_checkpoint,
        )

    def save(self, val_acc=1e9):
        """
        Save current network parameters, and update state variables.

        It also saves optimizer and scheduler parameters, if defined.
        """
        assert self.current_epoch > -1, "Current epoch is negative."
        val_acc = float(val_acc)
        if val_acc < self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = int(self.current_epoch)
        # Checkpoint data
        state = {
            "epoch": self.current_epoch,
            "step": self.current_step,
            "val_acc": val_acc,
            "best_epoch": self.best_epoch,
            "best_val_acc": self.best_val_acc,
            "state_dict": self.network.state_dict(),
        }
        torch.save(state, self.checkpoint_path(self.current_epoch))
        # Training checkpoint data
        if self.optimizer is None:
            return self
        state_training = {"optimizer": self.optimizer.state_dict()}
        if self.scheduler is not None:
            state_training.update({"scheduler": self.scheduler.state_dict()})
        torch.save(
            state_training,
            self.checkpoint_path(self.current_epoch, training_checkpoint=True),
        )
        return self

    def clean_saves(self):
        """
        Clean-up 'saves' folder.

        It keeps 'maximum_saves' best checkpoints, and the last checkpoint
        (required to continue training).
        """
        checkpoint_val_acc = []
        for f in sorted(os.listdir(self.saves_path)):
            if not os.path.isfile(os.path.join(self.saves_path, f)):
                continue
            f_split = f.split("-")
            if f_split[0] != "checkpoint":
                continue
            checkpoint = int(f_split[-1])
            checkpoint_data = torch.load(
                self.checkpoint_path(checkpoint),
                map_location=torch.device("cpu"),
                weights_only=True,
            )
            checkpoint_val_acc.append(
                (
                    checkpoint,
                    (checkpoint_data["val_acc"], checkpoint_data["epoch"]),
                )
            )
        if len(checkpoint_val_acc) - 1 > self.maximum_saves:
            checkpoints = [
                i[0]
                for i in sorted(checkpoint_val_acc[:-1], key=lambda x: x[1])
            ]
            for checkpoint in checkpoints[self.maximum_saves :]:
                checkpoint_path = self.checkpoint_path(checkpoint)
                if os.path.isfile(checkpoint_path):
                    os.remove(checkpoint_path)
                checkpoint_training_path = self.checkpoint_path(
                    checkpoint, training_checkpoint=True
                )
                if os.path.isfile(checkpoint_training_path):
                    os.remove(checkpoint_training_path)
        return self

    def restore(self, checkpoint=None, training=False, strict=True):
        """
        Restore a particular checkpoint.

        Parameter 'checkpoint' can be 'best', 'last', or an integer to restore a
        specific checkpoint. If 'training' is True, optimizer and scheduler
        parameters are also restored.
        """
        checkpoint_data = get_checkpoint_data(
            self.saves_path,
            checkpoint,
            training,
            self.current_epoch,
            self.device,
            self.checkpoint_n_digits,
        )
        if checkpoint_data is None:
            return self
        self.current_epoch = checkpoint_data["epoch"]
        self.current_step = checkpoint_data["step"]
        self.best_epoch = checkpoint_data["best_epoch"]
        self.best_val_acc = checkpoint_data["best_val_acc"]
        self.network.load_state_dict(checkpoint_data["state_dict"], strict)
        if training and self.optimizer is not None:
            checkpoint = checkpoint_data["epoch"]
            checkpoint_training_path = os.path.join(
                self.saves_path,
                (
                    "checkpoint_training-"
                    f"{checkpoint:0{self.checkpoint_n_digits}d}"
                ),
            )
            checkpoint_training_data = torch.load(
                os.path.join(self.saves_path, checkpoint_training_path),
                map_location=self.device,
                weights_only=True,
            )
            self.optimizer.load_state_dict(
                checkpoint_training_data["optimizer"]
            )
        if training and self.scheduler is not None:
            self.scheduler.load_state_dict(
                checkpoint_training_data["scheduler"]
            )
        print(
            f"Restored checkpoint: {self.current_epoch} "
            f"(best: {self.best_epoch})"
        )
        return self

    def get_result_path(self, name):
        """
        Return the path of an item 'name' located in 'results' folder.
        """
        return os.path.join(self.results_path, name)

    def get_new_result_path(self, prefix, extension, separator="-", n_digits=3):
        """
        Return the path of an item 'prefix.extension' located in 'results'
        folder.

        If this item already exists, it will be incremented, such as
        'prefix|separator|000.extension'.
        """
        i = 0
        path = os.path.join(
            self.results_path,
            f"{prefix}{separator}{i:0{n_digits}d}.{extension}",
        )
        while os.path.isfile(path):
            i += 1
            path = os.path.join(
                self.results_path,
                f"{prefix}{separator}{i:0{n_digits}d}.{extension}",
            )
        return path

    def clean_empty_folders(self, extra_folder_names=None):
        """
        Clean-up empty model folders located in 'logs', 'saves', and 'results'
        folders.
        """
        folder_names = ("logs", "saves", "results")
        if extra_folder_names is not None:
            folder_names += extra_folder_names
        for f_0 in os.listdir(self.project_path):
            if f_0 in folder_names:
                p_0 = os.path.join(self.project_path, f_0)
                for f_1 in os.listdir(p_0):
                    p_1 = os.path.join(p_0, f_1)
                    if os.path.isdir(p_1) and not os.listdir(p_1):
                        os.rmdir(p_1)
        return self

    def get_seed_rng(self):
        """
        Return a seed generator enforcing reproducible training sessions.
        """
        return _SeedGenerator(
            self.seed, self.current_epoch, self.checkpoint_n_digits
        )
