from copy import deepcopy
import math
import random
import torch
import numpy as np

from torch.utils.data import DataLoader
from ExpToolKit.models import BaseAE

from .utils import kwargs_parser, train_epoch, test, get_examples


class Denoiser:
    def __init__(self, model: BaseAE, **kwargs) -> None:
        """
        Initialize the Denoiser class.

        Parameters
        ----------
        model : BaseAE
            The autoencoder model to be used for denoising.
        **kwargs :
            Additional parameters to be passed to the specified model.

        Returns
        -------
        None
        """
        if not isinstance(model, BaseAE):
            raise TypeError("model must be an instance of BaseAE")
        self.model = deepcopy(model)

    def fit(self, dataloader: DataLoader, is_print: bool = True, **kwargs) -> None:
        """
        Train the model on the given dataset with noise added.

        Parameters
        ----------
        dataloader : DataLoader
            The DataLoader for the training data.
        is_print : bool, optional
            If True, print the training loss at each batch. Default is True.
        **kwargs :
            Additional parameters to be passed to the specified model.

            Including **random_seed**, **device**, **optimizer**, **noise_type**, **epoches**.

            Check default values in utils.py and const.py

        Returns
        -------
        float
            The average reconstruction loss of the model on the given dataset.
        """
        random_seed, device, optimizer, noise_type, epoches = kwargs_parser(
            self.model, kwargs
        )

        for epoch in range(1, epoches + 1):
            train_epoch(
                self.model,
                dataloader,
                random_seed,
                device,
                optimizer,
                noise_type,
                epoch,
                is_print,
                kwargs,
            )

        recon_loss = test(self.model, dataloader, device, noise_type, kwargs)
        return recon_loss

    def predict(self, dataloader: DataLoader, **kwargs):
        """
        Evaluate the model on the given dataset with noise and return the average reconstruction loss.

        Parameters
        ----------
        dataloader : DataLoader
            The DataLoader for the data to be predicted.
        **kwargs :
            Additional parameters to be passed to the specified model.

            Including **random_seed**, **device**, **optimizer**, **noise_type**, **epoches**.

            Check default values in utils.py and const.py

        Returns
        -------
        float
            The average reconstruction loss of the model on the given dataset.
        """
        random_seed, device, optimizer, noise_type, epoches = kwargs_parser(
            self.model, kwargs
        )
        del random_seed, optimizer, epoches
        return test(self.model, dataloader, device, noise_type, kwargs)

    def evaluate(self, dataloader: DataLoader, **kwargs):
        """
        Evaluate the model on the given dataset with noise and return the average reconstruction loss.

        Parameters
        ----------
        dataloader : DataLoader
            The DataLoader for the data to be predicted.
        **kwargs :
            Additional parameters to be passed to the specified model.

            Including **random_seed**, **device**, **optimizer**, **noise_type**, **epoches**.

            Check default values in utils.py and const.py

        Returns
        -------
        float
            The average reconstruction loss of the model on the given dataset.
        """
        random_seed, device, optimizer, noise_type, epoches = kwargs_parser(
            self.model, kwargs
        )
        del random_seed, optimizer, epoches
        return test(self.model, dataloader, device, noise_type, kwargs)

    def get_model(self) -> BaseAE:
        """
        Get the model of the Denoiser.

        Returns
        -------
        BaseAE
            The model of the Denoiser.
        """
        return self.model

    def get_example(
        self, dataloader: DataLoader, indexes: list[int], **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the original and denoised examples from the given dataloader at the given indexes.

        Parameters
        ----------
        dataloader : DataLoader
            The DataLoader for the data to be processed.
        indexes : list[int]
            The indexes of the examples to be returned.
        **kwargs :
            Additional parameters to be passed to the specified model.

            Including **random_seed**, **device**, **optimizer**, **noise_type**, **epoches**.

            Check default values in utils.py and const.py

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the original examples and the denoised examples.
        """
        random_seed, optimizer, device, noise_type, epoches = kwargs_parser(
            self.model, kwargs
        )
        del random_seed, optimizer, epoches

        return get_examples(self.model, dataloader, indexes, noise_type, kwargs)
