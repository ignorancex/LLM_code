"""This module implements dataset loading for the USCG network."""
import logging
from typing import Tuple
import numpy as np
import torch
from ..utils import get_hits_from_clear_images
from dunedn import PACKAGE
from dunedn.utils.utils import median_subtraction

logger = logging.getLogger(PACKAGE + ".gcnn")


class BaseUscgDataset(torch.utils.data.Dataset):
    """Loads the dataset for CNN and GCNN networks."""

    def __init__(
        self,
        dataset_type: str,
        task: str,
        channel: str,
        dsetup: dict,
        batch_size: int,
    ):
        """
        Parameters
        ----------
        dataset_type: str
            Available options train | val | test
        task: str
            Available options dn | roi.
        channel: str
            Available options induction | collection
        dsetup: dict
            The dataset settings dictionary.
        batch_size: int
            The number of examples to be batched.
        """
        self.dataset_type = dataset_type
        self.task = task
        self.channel = channel
        self.dsetup = dsetup
        self.batch_size = int(batch_size)

        self.training = self.dataset_type == "train"
        self.threshold = dsetup["threshold"]

    def __len__(self):
        return len(self.noisy)


class UscgDataset(BaseUscgDataset):
    """Loads the dataset for CNN and GCNN networks."""

    def __init__(
        self,
        dataset_type: str,
        task: str,
        channel: str,
        dsetup: dict,
        batch_size: int,
    ):
        """
        Parameters
        ----------
        dataset_type: str
            Available options train | val | test
        task: str
            Available options dn | roi.
        channel: str
            Available options induction | collection
        dsetup: dict
            The dataset settings dictionary.
        batch_size: int
            The number of examples to be batched.
        """
        super().__init__(
            dataset_type,
            task,
            channel,
            dsetup,
            batch_size,
        )

        self.data_folder = self.dsetup["data_folder"] / dataset_type
        self.planes_folder = self.data_folder / "planes"

        self.noisy, self.clear = self.get_planes_from_setup()

    def get_planes_from_setup(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get planes from folder pointed by `dsetup`.

        Returns
        -------
        noisy: torch.Tensor
            The noisy planes, of shape=(N,1,H,W).
        clear: torch.Tensor
            The clear planes, of shape=(N,1,H,W).
        """
        fname = self.planes_folder / f"{self.channel}_noisy.npy"
        noisy = np.load(fname)
        noisy = median_subtraction(noisy)

        fname = self.planes_folder / f"{self.channel}_clear.npy"
        clear = np.load(fname)

        if self.task == "roi":
            clear = get_hits_from_clear_images(clear, self.threshold)
            self.balance_ratio = np.count_nonzero(clear) / clear.size()

        noisy = torch.Tensor(noisy)
        clear = torch.Tensor(clear)
        return noisy, clear

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        noisy: torch.Tensor
            A single noisy example, of shape=(1,H,W).
        clear: torch.Tensor
            A single clear example, of shape=(1,H,W).
        """
        return self.noisy[index], self.clear[index]


class UscgPlanesDataset(BaseUscgDataset):
    """Loads planes in dataset form for GcnnNet network inference."""

    def __init__(
        self,
        noisy: np.ndarray,
        task: str,
        channel: str,
        dsetup: dict,
        batch_size: int,
    ):
        """
        Parameters
        ----------
        noisy: np.ndarray
            The noisy planes for inference.
        task: str
            Available options dn | roi.
        channel: str
            Available options induction | collection
        dsetup: dict
            The dataset settings dictionary.
        batch_size: int
            The number of examples to be batched.
        """
        noisy = noisy.astype(np.float32)
        self.noisy = median_subtraction(noisy)
        super().__init__(
            "test",
            task,
            channel,
            dsetup,
            batch_size,
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns
        -------
        noisy: torch.Tensor
            A single noisy example, of shape=(1,H,W).
        None
            dummy output for labels.
        """
        return self.noisy[index], 0
