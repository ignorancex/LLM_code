import pathlib
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.thermometer_encoding import data_to_therm_numpy

# %%

# =============================================================================
# Function adapted from file:
#   https://github.com/Shmoo137/Interpretable-Phase-Classification/blob/master/Influence_Functions_LL-CDW/data_loader.py
# =============================================================================

# =============================================================================
# Generate PyTorch dataset (it transforms array of ndarrays to torch tensor).
# Overloading of '__selt__'. and '__getitem__' is required by the parent class.
# =============================================================================


class NumpyToPyTorch_DataLoader(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = torch.from_numpy(X).float()  # image
        self.Y = torch.from_numpy(Y).long()  # label for classification
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        label = self.Y[index]
        img = self.X[index]

        if self.transform:
            img = self.transform(img)

        return img, label


class Importer(object):
    """
    Parameters
    ----------
    dataset_folder_path : pathlib.PosixPath
        A pathlib Path object pointing to the directory in which you have
        stored the pickled train/validation/test datasets.
    batch : int
        Specifies the default batch size.
    device : torch.device
        Allows for enabling of memory pin for CUDA-enabled computers.
        If is not None, then if enables pin_memory option for CUDA devices.
        Default is None.
    noise_pow : float, optional
        Should the random Gaussian noise be added to each training example,
        with μ=0, σ=noise_pow. If 'None', then no noise is applied.
        Default is None.

    Returns
    -------
    Nothing, just states the internal parameters state.

    """

    def __init__(
        self,
        dataset_folder_path,
        batch,
        device=None,
        noise_pow=None,
        therm_levels=None,
        perturbative_ds=None,
    ):
        """
        Parameters
        ----------
        dataset_folder_path : pathlib.PosixPath
            A pathlib Path object pointing to the directory in which you have
            stored the pickled train/validation/test datasets.
        batch : int
            Specifies the default batch size.
        device : torch.device
            Allows for enabling of memory pin for CUDA-enabled computers.
            If is not None, then if enables pin_memory option for CUDA devices.
            Default is None.
        noise_pow : float, optional
            Should the random Gaussian noise be added to each training example,
            with μ=0, σ=noise_pow. If 'None', then no noise is applied.
            Default is None.
        therm_levels : int, optional
            IF you decide to use thermometer encoding, this variable states
            how many thermometer levels are used
        perturbative_ds : list, optional
            IF you decide to use perturbative approximation for training
            (datasets with little disorder), this should be the list where
            you give paths to them.

        Returns
        -------
        Nothing, just states the internal parameters state.

        """

        self.batch = batch
        self.datasets_path = dataset_folder_path
        self._train_path = dataset_folder_path.joinpath("training_set.pickle")
        self._val_path = dataset_folder_path.joinpath("validation_set.pickle")
        self._test_path = dataset_folder_path.joinpath("test_set.pickle")
        if device is not None:
            self.mem_pin_bool = True if device.type == "cuda" else False
        else:
            self.mem_pin_bool = False
        self.noise = noise_pow
        self.therm_levels = therm_levels
        self.perturbative_ds = perturbative_ds

    def get_train_loader(
        self, batch_size=None, shuffle=True, save_mask=False, seed=2137
    ):
        """
        A function for preparation of PyTorch DataLoader class instance with
        training dataset loaded.

        Parameters
        ----------
        batch_size : int, optional
            Custom batch size for training dataset loader.
            If None, Importer instance's default value is used.
            The default is None.
        shuffle : bool, optional
            Should the training data be shuffled on import. The default is True.
        save_mask : bool, optional
            Should the shuffle mask be saved to hard drive apart from it being
            saved as variable in Importer instance. The default is False.
        seed : int, optional
            What seed should the random number generator use for shuffling.
            The default is 2137.

        Returns
        -------
        train_loader : torch.utils.data.dataloader.DataLoader
            DataLoader instance fed with training dataset.

        """

        if batch_size is None:
            batch_size = self.batch

        with open(self._train_path, "rb") as f:
            train_dict = pickle.load(f)
            self.M = train_dict["data"][0].shape[0]

        if self.perturbative_ds is not None:
            perturbative = []
            for ds_path in self.perturbative_ds:
                with open(ds_path.joinpath("training_set.pickle"), "rb") as f:
                    perturbative.append(pickle.load(f))

        train_keys = list(train_dict.keys())

        for key in train_keys:
            if isinstance(train_dict[key], list):
                train_dict[key] = np.array(train_dict[key])

        train_data = train_dict.copy()
        if self.perturbative_ds is not None:
            for pert in perturbative:
                for key in train_keys:
                    if isinstance(pert[key], list):
                        pert[key] = np.array(pert[key])
                    train_data[key] = np.append(train_data[key], pert[key], axis=0)

        train_samples_num = train_data[train_keys[0]].shape[0]

        if shuffle:
            # Shuffling ordered data, but preserving the mask (since we want to remember for which 'v' the datapoint was calculated)
            mask = np.arange(train_samples_num)
            np.random.seed(seed)
            np.random.shuffle(mask)

            # Saving the mask in the object's variables to retrieve original indices afterwards
            self.train_mask = mask
            if save_mask:
                with open(
                    self.datasets_path.joinpath("train_set_mask.pickle"), "wb"
                ) as f:
                    pickle.dump(mask, f)

            masked_train_data = train_data["data"][mask]
            masked_train_labels = train_data["labels"][mask]

        else:
            masked_train_data = train_data["data"]
            masked_train_labels = train_data["labels"]

        if self.noise is not None:
            noise_tensor = np.random.normal(0, self.noise, size=masked_train_data.shape)
            masked_train_data = np.add(masked_train_data, noise_tensor)

        masked_train_data = masked_train_data.reshape([-1, 1, self.M, self.M])

        if self.therm_levels is not None:
            masked_train_data = data_to_therm_numpy(
                x=masked_train_data, levels=self.therm_levels
            )

        train_set = NumpyToPyTorch_DataLoader(masked_train_data, masked_train_labels)

        self.train_W_tab = train_data["W"]
        self.train_v_tab = train_data["v"]

        if batch_size == -1:
            batch_size = train_samples_num

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.mem_pin_bool,  # CUDA only, this lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer from CPU to GPU during training
        )
        return train_loader

    def get_val_loader(self, batch_size=None):
        """
        A function for preparation of PyTorch DataLoader class instance with
        validation dataset loaded.

        Parameters
        ----------
        batch_size : int, optional
            Custom batch size for validation dataset loader.
            If None, Importer instance's default value is used.
            The default is None.

        Returns
        -------
        val_loader : torch.utils.data.dataloader.DataLoader
            DataLoader instance fed with validation dataset.

        """

        if batch_size is None:
            batch_size = self.batch

        with open(self._val_path, "rb") as f:
            val_dict = pickle.load(f)
            self.M = val_dict["data"][0].shape[0]

        val_keys = list(val_dict.keys())

        for key in val_keys:
            if isinstance(val_dict[key], list):
                val_dict[key] = np.array(val_dict[key])

        if self.perturbative_ds is not None:
            perturbative = []
            for ds_path in self.perturbative_ds:
                with open(ds_path.joinpath("validation_set.pickle"), "rb") as f:
                    perturbative.append(pickle.load(f))

        val_data = val_dict.copy()
        if self.perturbative_ds is not None:
            for pert in perturbative:
                for key in val_keys:
                    if isinstance(pert[key], list):
                        pert[key] = np.array(pert[key])
                    val_data[key] = np.append(val_data[key], pert[key], axis=0)

        data = val_data["data"]
        data = data.reshape([-1, 1, self.M, self.M])

        if self.therm_levels is not None:
            data = data_to_therm_numpy(x=data, levels=self.therm_levels)

        val_set = NumpyToPyTorch_DataLoader(data, val_data["labels"])

        self.val_W_tab = val_data["W"]
        self.val_v_tab = val_data["v"]

        if batch_size == -1:
            batch_size = val_data["data"].shape[0]

        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.mem_pin_bool,
        )
        return val_loader

    def get_test_loader(self, batch_size=None, realization=None, subset=None):
        """
        A function for preparation of PyTorch DataLoader class instance with
        test dataset loaded.

        Parameters
        ----------
        batch_size : int, optional
            Custom batch size for test dataset loader.
            If None, Importer instance's default value is used.
            The default is None.
        realization : int, optional
            IF not None, then this gets appended to set file name.
            An option for testing on various disorder realizations.
        subset : list, optional
            If you want to use only a subset of the dataset, you can specify
            it here. It should be a list of integers, where each integer
            corresponds to a data point index. The default is None.

        Returns
        -------
        test_loader : torch.utils.data.dataloader.DataLoader
            DataLoader instance fed with test dataset.

        """

        if batch_size is None:
            batch_size = self.batch

        if realization is None:
            with open(self._test_path, "rb") as f:
                test_dict = pickle.load(f)
                self.M = test_dict["data"][0].shape[0]
        else:
            with open(
                self.datasets_path.joinpath(f"test_{realization}_set.pickle"), "rb"
            ) as f:
                test_dict = pickle.load(f)
                self.M = test_dict["data"][0].shape[0]

        test_keys = list(test_dict.keys())

        for key in test_keys:
            if isinstance(test_dict[key], list):
                test_dict[key] = np.array(test_dict[key])

        if subset is not None:
            test_dict["data"] = test_dict["data"][subset].copy()
            test_dict["labels"] = test_dict["labels"][subset].copy()
            test_dict["W"] = test_dict["W"][subset].copy()
            test_dict["v"] = test_dict["v"][subset].copy()

        data = test_dict["data"]

        data = data.reshape([-1, 1, self.M, self.M])

        if self.therm_levels is not None:
            data = data_to_therm_numpy(x=data, levels=self.therm_levels)

        test_set = NumpyToPyTorch_DataLoader(data, test_dict["labels"])

        self.W_tab = test_dict["W"]
        self.v_tab = test_dict["v"]
        self.test_data = test_dict["data"]
        self.test_labels = test_dict["labels"]

        if batch_size == -1:
            batch_size = test_dict["data"].shape[0]

        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.mem_pin_bool,
        )
        return test_loader
