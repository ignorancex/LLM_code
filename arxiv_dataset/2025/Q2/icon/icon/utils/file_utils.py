import os
import torch
import logging
from pathlib import Path
from typing import Optional, Union, Literal, Dict


def str2path(path: str) -> Path:
    return Path(os.path.expanduser(path)).absolute()


def mkdir(
    path: Union[str, Path],
    parents: Optional[bool] = False,
    exist_ok: Optional[bool] = False
) -> Path:
    path = str2path(path)
    path.mkdir(parents=parents, exist_ok=exist_ok)
    return path


def create_logger(logging_dir: Union[str, Path, None] = None) -> logging.Logger:
    if logging_dir is not None:
        logging_dir = mkdir(logging_dir, True, True)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(str(logging_dir.joinpath("train.log")))
            ]
        )
    logger = logging.getLogger(__name__)
    return logger


class CheckpointManager:

    def __init__(
        self,
        save_dir: Union[str, Path],
        val_freq: Optional[int] = 1,
        topk: Optional[int] = 1,
        mode: Literal['max', 'min'] = 'min',
        save_last: Optional[bool] = True
    ) -> None:
        """
        Args:
            save_dir (str): directory used for saving checkpoints.
            val_freq (int, optional): validation frequency.
            topk (int, optional): only checkpoints with top-k performance would be saved.
            mode (str, optional): checkpoint saving mode.
                - 'max': save the checkpoints with maximum metric values.
                - 'min': save the checkpoints with minimum metric values.
            save_last (bool, optional): if True, save the checkpoint after the last epoch.
        """
        assert topk >= 0, "@topk must be a non-negative integer!"
        assert mode in ['max', 'min']
        self.save_dir = mkdir(save_dir, True, True)
        self.val_freq = val_freq
        self.topk = topk
        self.mode = mode
        self.save_last = save_last
        self.global_step = 1
        self.epochs = list()
        self.metrics = list()
        self.state_dicts = list()

    def update(self, metric: float, state_dict: Dict) -> None:
        self.epochs.append(self.global_step * self.val_freq)
        self.metrics.append(metric)
        self.state_dicts.append(state_dict)
        self.global_step += 1

    def save_topk(self) -> None:
        if len(self.state_dicts) == 0:
            print("WARNING: No data stored in the buffer. Calling this function would save nothing.")
        else:
            last_epoch, last_metric, last_state_dict = self.epochs[-1], self.metrics[-1], self.state_dicts[-1]
            reverse = self.mode == 'max'
            metrics, epochs, state_dicts = zip(*sorted(zip(self.metrics, self.epochs, self.state_dicts), reverse=reverse))
            if self.topk > 0:
                epochs = list(epochs)
                metrics = list(metrics)
                state_dicts = list(state_dicts)
                if self.topk > len(state_dicts):
                    print(f"WARNING: only {len(state_dicts)} checkpoints would be saved.")
                else:
                    epochs = epochs[:self.topk]
                    metrics = metrics[:self.topk]
                    state_dicts = state_dicts[:self.topk]
                if self.save_last:
                    epochs.append(last_epoch)
                    metrics.append(last_metric)
                    state_dicts.append(last_state_dict)
                for epoch, metric, state_dict in zip(epochs, metrics, state_dicts):
                    torch.save(state_dict, str(self.save_dir.joinpath(f"{epoch}_{round(metric, 5)}.pth")))
        
    def save(self, state_dict: Dict) -> None:
        torch.save(state_dict, str(self.save_dir.joinpath("checkpoint.pth")))

    def clear(self) -> None:
        self.global_step = 1
        self.metrics = list()
        self.state_dicts = list()


