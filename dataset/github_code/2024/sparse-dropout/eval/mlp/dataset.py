from pathlib import Path

import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST


def load_data_module(name: str, **kwargs):
    match name:
        case "mnist":
            return MNISTDataModule(**kwargs)
        case "fashion_mnist":
            return FashionMNISTDataModule(**kwargs)
        case "cifar10":
            return CIFAR10DataModule(**kwargs)
        case _:
            raise ValueError(f"Unknown data module: {name}")


class VisionDataModule(L.LightningDataModule):
    num_classes: int
    img_size: tuple[int, int]

    def __init__(
        self,
        builder,
        train_batch_size: int,
        train_size: int,
        val_size: int,
        val_batch_size: int | None = None,
    ):
        super().__init__()
        self.builder = builder
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size or train_batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.img_size, antialias=True),
            ]
        )

    @property
    def train_samlpe(self):
        return next(iter(self.train_dataloader()))

    def prepare_data(self):
        self.full_dataset = self.builder(self.transform)

    def setup(self, stage: str):
        if stage == "fit":
            indices = np.random.choice(
                len(self.full_dataset), self.train_size + self.val_size, replace=False
            ).tolist()
            self.train_subset = Subset(self.full_dataset, indices[: self.train_size])
            self.val_subset = Subset(self.full_dataset, indices[self.train_size :])
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_subset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_subset,
            batch_size=self.val_batch_size,
            num_workers=4,
        )


class MNISTDataModule(VisionDataModule):
    num_classes = 10
    img_size = (32, 32)

    def __init__(
        self,
        train_batch_size: int,
        train_size: int,
        val_size: int,
        val_batch_size: int | None = None,
    ):
        super().__init__(
            lambda transform: MNIST(
                str(Path("data", "MNIST")),
                train=True,
                download=True,
                transform=transform,
            ),
            train_batch_size,
            train_size,
            val_size,
            val_batch_size,
        )


class FashionMNISTDataModule(VisionDataModule):
    num_classes = 10
    img_size = (32, 32)

    def __init__(
        self,
        train_batch_size: int,
        train_size: int,
        val_size: int,
        val_batch_size: int | None = None,
    ):
        super().__init__(
            lambda transform: FashionMNIST(
                str(Path("data", "Fashion_MNIST")),
                train=True,
                download=True,
                transform=transform,
            ),
            train_batch_size,
            train_size,
            val_size,
            val_batch_size,
        )


class CIFAR10DataModule(VisionDataModule):
    num_classes = 10
    img_size = (32, 32)

    def __init__(
        self,
        train_batch_size: int,
        train_size: int,
        val_size: int,
        val_batch_size: int | None = None,
    ):
        super().__init__(
            lambda transform: CIFAR10(
                str(Path("data", "CIFAR10")),
                train=True,
                download=True,
                transform=transform,
            ),
            train_batch_size,
            train_size,
            val_size,
            val_batch_size,
        )
