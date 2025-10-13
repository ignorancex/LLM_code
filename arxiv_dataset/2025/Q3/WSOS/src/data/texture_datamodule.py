from typing import Any, Dict, Optional, Tuple, List
import os

import torchvision
from PIL import Image
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision.transforms import transforms

from src.utils import extract_patch


class TextureDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
            self,
            images_number: int,
            texture_dir: str,
            data_dir: str = "data/",
            im_size: List[int] = [90, 90],
            train_val_test_split: Tuple[float, float, float] = (.8, .1, .1),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.images_number = images_number
        self.texture_dir = texture_dir
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.texture_sources: str = None
        self.batch_size_per_device = batch_size
        self.im_size = im_size
        self.x = None
        self.y = None

    def set_logger(self, logger):
        self.logger = logger
        
    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        self.texture_sources, self.texture_sources_labels = self.load_textures()
        self.process_bg_images()

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            print(self.x.shape, self.y.shape)
            dataset = TensorDataset(self.x, self.y)
            total_samples = len(dataset)
            train_percentage, val_percentage, _ = self.hparams.train_val_test_split
            train_size = int(train_percentage * total_samples)
            val_size = int(val_percentage * total_samples)
            test_size = total_samples - train_size - val_size

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=[train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def load_textures(self):
        """ Load source textures

        Load the texture images that will be used as a background for the MNIST
        digits.

        Args
        ----
        root: str
            Path to the folder with texture images.

        Returns
        -------
        textures: torch.Tensor
            A tensor containing texture images.
        ids: list[int]
            A list with the id of each texture image.

        Notes
        -----
        The name of each texture image file is supposed to have a
        unique numeric id. The id of the `i`-th image is saved in `ids[i]`.
        """

        textures, ids = [], []

        # Define a transform to convert PIL images to tensors
        transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        for filename in os.listdir(self.texture_dir):
            path = os.path.join(self.texture_dir, filename)
            if os.path.isfile(path) and filename.lower().endswith('.tif'):
                # Read the TIFF image using PIL
                with Image.open(path) as im:
                    # Convert the image to grayscale
                    im = im.convert('L')
                    # Convert PIL image to a tensor
                    tensor_image = torchvision.transforms.functional.pil_to_tensor(im)  # Add a batch dimension
                    textures.append(tensor_image)
                    ids.append(int(os.path.splitext(filename)[0][1:]))  # Extract id from filename

        # Stack the tensor images along a new dimension to create a single tensor
        textures = torch.stack(textures)
        ids = torch.tensor(ids)

        return textures, ids

    def process_bg_images(self):
        textures = torch.zeros((self.images_number, 1, self.im_size[0], self.im_size[1]),
                               dtype=torch.uint8)
        texture_labels = torch.zeros(self.images_number, dtype=torch.uint8)
        for i in range(self.images_number):
            associated_texture_label = torch.randint(0, len(self.texture_sources_labels), size=(1,))[0]
            textures[i] = extract_patch(self.texture_sources[associated_texture_label], self.im_size)
            texture_labels[i] = associated_texture_label
        self.x = textures
        self.y = texture_labels
