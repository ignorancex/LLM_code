import numpy as np
import torch
import pytorch_lightning as pl
from typing import Optional, Callable


class BaseDataset(torch.utils.data.Dataset):
    """
    Base class for custom datasets.

    Args:
        on_after_load_sample (Optional[Callable]): A callable function applied to each sample after loading.
    """
    
    def __init__(self, on_after_load_sample: Optional[Callable] = None):
        self._on_after_load_sample = on_after_load_sample

    def __getitem__(self, idx):
        """
        Abstract method to retrieve a data sample.

        Args:
            idx: Index of the sample.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __len__(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _process_sample(self, sample):
        """
        Processes a sample after loading.

        Args:
            sample (dict): The data sample to process.

        Returns:
            dict: The processed sample.
        """
        sample['x'] = sample['xi'] # Network will take input 'x', simulator returns 'xi'
        if self._on_after_load_sample is not None:
            sample = self._on_after_load_sample(sample)
        return sample
    

class StoredDataset(BaseDataset):
    """
    Dataset for pre-stored data.

    Args:
        data (dict): The data stored as a dictionary of arrays.
        on_after_load_sample (Optional[Callable]): A callable function applied to each sample after loading.
    """
    
    def __init__(self, data, on_after_load_sample: Optional[Callable] = None):
        super().__init__(on_after_load_sample)
        self._data = data

    def __len__(self):
        return len(self._data[list(self._data.keys())[0]])

    def __getitem__(self, idx):
        sample = {k: v[idx] for k, v in self._data.items()}
        return self._process_sample(sample)
    

class OnTheFlyDataset(BaseDataset):
    """
    Dataset that generates data on-the-fly using a simulator.

    Args:
        simulator: A simulator object with a `sample` method to generate data.
        Nsims_per_epoch (int): Number of simulations per epoch.
        on_after_load_sample (Optional[Callable]): A callable function applied to each sample after generation.
    """
    def __init__(self, simulator, Nsims_per_epoch: int, on_after_load_sample: Optional[Callable] = None):
        super().__init__(on_after_load_sample)
        self.simulator = simulator
        self.Nsims_per_epoch = Nsims_per_epoch

    def __len__(self):
        return self.Nsims_per_epoch

    def __getitem__(self, idx):
        sample = self.simulator.sample(Nsims=1)
        sample = {k: v[0] for k, v in sample.items()}
        return self._process_sample(sample)
    

class BaseDataModule(pl.LightningDataModule):
    """
    Base class for data modules in PyTorch Lightning.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        on_after_load_sample (Optional[Callable]): A callable function applied to each sample after loading.
        val_fraction (float): Fraction of data to use for validation.
        test_fraction (float): Fraction of data to use for testing.
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        on_after_load_sample: Optional[Callable] = None,
        val_fraction: float = 0.2,
        test_fraction: float = 0.1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.on_after_load_sample = on_after_load_sample
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction

    def setup(self, stage: Optional[str] = None):
        """
        Abstract method to set up datasets for different stages.

        Args:
            stage (Optional[str]): Stage of setup ('train', 'test', etc.).

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _create_dataloader(self, dataset, shuffle: bool):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, shuffle=False)


class StoredDataModule(BaseDataModule):
    """
    Data module for pre-stored data.

    Args:
        data (dict): The data stored as a dictionary of arrays.
        **kwargs: Additional arguments passed to BaseDataModule.
    """
    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self.data = data

    def setup(self, stage: Optional[str] = None):
        dataset = StoredDataset(self.data)
        total_samples = len(dataset)
        n_val = int(self.val_fraction * total_samples)
        n_test = int(self.test_fraction * total_samples)
        n_train = total_samples - n_val - n_test
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test]
        )
        # Apply transformation only to train dataset
        self.train_dataset.dataset._on_after_load_sample = self.on_after_load_sample


class OnTheFlyDataModule(BaseDataModule):
    """
    Data module that generates data on-the-fly using a simulator.

    Args:
        simulator: A simulator object with a `sample` method to generate data.
        Nsims_per_epoch (int): Number of simulations per epoch.
        **kwargs: Additional arguments passed to BaseDataModule.
    """
    
    def __init__(self, simulator, Nsims_per_epoch: int = 10000, **kwargs):
        super().__init__(**kwargs)
        self.simulator = simulator
        self.Nsims_per_epoch = Nsims_per_epoch

    def setup(self, stage: Optional[str] = None):
        n_val = int(self.val_fraction * self.Nsims_per_epoch)
        n_test = int(self.test_fraction * self.Nsims_per_epoch)
        n_train = self.Nsims_per_epoch - n_val - n_test

        # Training dataset with on-the-fly data generation
        self.train_dataset = OnTheFlyDataset(
            simulator=self.simulator,
            Nsims_per_epoch=n_train,
            on_after_load_sample=self.on_after_load_sample,
        )

        # Validation and test datasets with fixed data
        val_data = self.simulator.sample(Nsims=n_val)
        test_data = self.simulator.sample(Nsims=n_test)
        self.val_dataset = StoredDataset(val_data)
        self.test_dataset = StoredDataset(test_data)
