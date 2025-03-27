"""This module implements dataset loading for the CNN and GCNN networks."""
from abc import ABC, abstractmethod
import logging
from typing import Tuple
import numpy as np
import torch
from ..utils import get_hits_from_clear_images
from .gcnn_net_utils import Converter
from dunedn import PACKAGE
from dunedn.utils.utils import median_subtraction

logger = logging.getLogger(PACKAGE + ".gcnn")


class BaseGcnnDataset(torch.utils.data.Dataset, ABC):
    """Loads the dataset for CNN and GCNN networks."""

    def __init__(
        self,
        dataset_type: str,
        task: str = "dn",
        channel: str = "collection",
        dsetup: dict = None,
        batch_size: int = 128,
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
        self.crop_size = self.dsetup["crop_size"]
        self.threshold = dsetup["threshold"]

    def __len__(self):
        return len(self.noisy)

    @abstractmethod
    def __getitem__(self, index):
        pass


class GcnnDataset(BaseGcnnDataset):
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
        super().__init__(dataset_type, task, channel, dsetup, batch_size)

        self.data_folder = self.dsetup["data_folder"] / dataset_type
        self.crops_folder = self.data_folder / "crops"
        self.planes_folder = self.data_folder / "planes"

        crop_edge = self.dsetup["crop_edge"]
        pct = self.dsetup["pct"]

        # if dataset_type is training, load crops. Load planes otherwise.
        if self.training:
            fname = self.crops_folder / f"{channel}_clear_{crop_edge}_{pct}.npy"
            clear = np.load(fname)

            fname = self.crops_folder / f"{channel}_noisy_{crop_edge}_{pct}.npy"
            # median subtraction is made on a plane basis: crops are already
            # normalized during preprocessing stage
            noisy = np.load(fname)
        else:
            fname = self.planes_folder / f"{channel}_clear.npy"
            clear = np.load(fname)

            fname = self.planes_folder / f"{channel}_noisy.npy"
            noisy = np.load(fname)
            noisy = median_subtraction(noisy)
            self.converter = Converter(self.crop_size)

        if self.task == "roi":
            clear = get_hits_from_clear_images(clear, self.threshold)
            self.balance_ratio = np.count_nonzero(clear) / clear.size()

        self.noisy = torch.Tensor(noisy)
        self.clear = torch.Tensor(clear)

    def to_crops(self):
        """Converts planes into crops.

        Note
        ----

        This method should not be called when training.
        """
        if self.training:
            logger.error("`to_crops()` method should not be called when training")

        self.noisy = self.converter.planes2tiles(self.noisy)
        self.clear = self.converter.planes2tiles(self.clear)

    def to_planes(self):
        """Converts crops into planes."""
        if self.training:
            logger.error("`to_planes()` method should not be called when training")
        self.noisy = self.converter.tiles2planes(self.noisy)
        self.clear = self.converter.tiles2planes(self.clear)

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


class GcnnPlanesDataset(BaseGcnnDataset):
    """Loads the dataset for CNN and GCNN networks."""

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
        super().__init__("test", task, channel, dsetup, batch_size)

        self.converter = Converter(self.crop_size)

        noisy = median_subtraction(noisy)
        self.noisy = torch.Tensor(noisy)

    def to_crops(self):
        """Converts planes into crops.

        Note
        ----

        This method should not be called when training.
        """
        if self.training:
            logger.error("`to_crops()` method should not be called when training")

        self.noisy = self.converter.planes2tiles(self.noisy)

    def to_planes(self):
        """Converts crops into planes."""
        if self.training:
            logger.error("`to_planes()` method should not be called when training")
        self.noisy = self.converter.tiles2planes(self.noisy)

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
