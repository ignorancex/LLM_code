import numpy as np
import torch

from .loss import get_loss_func
from .pde_residual import get_pde_residual


class Observation:
    def __init__(self, config, dataset_name):
        self.dataset_name = dataset_name
        self.config = config
        self.type = config["type"]
        self.loss_type = config["loss_type"]
        self.loss_func = get_loss_func(self.loss_type)

    def init(self, ground_truth):
        raise NotImplementedError

    def get_observation_loss(self, x_pred):
        raise NotImplementedError

    def _calculate_loss(self, pred, gt, n_obs, mask=None):
        """Calculate the loss for a single field.

        Args:
            pred (torch.Tensor): Predicted field
            gt (torch.Tensor): Ground truth field
            mask (torch.Tensor): Observation mask
            n_obs (int): Number of observation points

        Returns:
            torch.Tensor: Observation loss for the field
        """
        if mask is None:
            loss = self.loss_func(pred - gt, n_obs)
        else:
            loss = self.loss_func((pred - gt) * mask, n_obs)
        return loss


class FullObservation(Observation):
    """Class to handle full observations for PDE solving.

    This class calculates losses using the complete field without masking.
    """

    def init(self, ground_truth, normalizer=None):
        """Initialize the full observation handler.

        Args:
            ground_truth (torch.Tensor): Ground truth data
            normalizer (optional): Normalizer object for data normalization
        """
        self.device = ground_truth.device
        self.resolution = ground_truth.shape[-1]
        self.n_channels = ground_truth.shape[1]
        self.ground_truth = ground_truth

        # Full observation means all points are observed
        self.known_indices = torch.ones(self.n_channels, device=self.device) * (self.resolution**2)

        # Add normalization support
        self.to_normalize = self.config["normalize"]
        if self.to_normalize:
            self.normalizer = normalizer
            self.ground_truth = self.normalizer.normalize(self.ground_truth)

    def get_observation_loss(self, x_pred):
        if self.to_normalize:
            x_pred = self.normalizer.normalize(x_pred)
        return self._calculate_loss(x_pred, self.ground_truth, self.known_indices)


class SparseObservation(Observation):
    """Class to handle sparse observations for PDE solving.

    This class manages the observation masks and calculates observation losses
    for both channels in the PDE system.
    """

    def init(self, ground_truth, normalizer=None):
        """Initialize the sparse observation handler.

        Args:
            data (torch.Tensor): Ground truth data
            config (dict): Configuration dictionary containing observation settings
        """
        self.device = ground_truth.device
        self.resolution = ground_truth.shape[-1]
        self.n_channels = ground_truth.shape[1]
        self.ground_truth = ground_truth

        self.masks = []
        self.n_obs_list = []
        for n_obs in self.config["known_indices"]:
            if 0 < n_obs < 1:
                n_obs = int(n_obs * self.resolution**2)
            mask = self.generate_random_mask(n_obs)
            self.masks.append(mask)
            self.n_obs_list.append(max(n_obs, 1))
        self.masks = torch.stack(self.masks, dim=0)
        self.known_indices = torch.tensor(self.n_obs_list, device=self.device).view(1, self.n_channels)

        self.to_normalize = self.config["normalize"]
        if self.to_normalize:
            self.normalizer = normalizer
            self.ground_truth = self.normalizer.normalize(self.ground_truth)

        self.interpolation_mode = None

    def generate_random_mask(self, k):
        """Generate a random binary mask with k ones.

        Args:
            k (int): Number of observation points

        Returns:
            torch.Tensor: Binary mask of shape [resolution, resolution]
        """
        indices = np.random.choice(self.resolution**2, k, replace=False)
        indices_2d = np.unravel_index(indices, (self.resolution, self.resolution))

        mask = torch.zeros((self.resolution, self.resolution), device=self.device)
        mask[indices_2d] = 1
        return mask

    def get_observation_loss(self, x_pred):
        if self.to_normalize:
            x_pred = self.normalizer.normalize(x_pred)

        if self.interpolation_mode is not None and x_pred.shape[-1] != self.resolution:
            x_pred = torch.nn.functional.interpolate(x_pred, size=(self.resolution, self.resolution), mode=self.interpolation_mode, align_corners=True)

        return self._calculate_loss(x_pred, self.ground_truth, self.known_indices, self.masks)


class PDEObservation(Observation):
    """Class to handle PDE loss calculations for the solver.

    This class manages the calculation of PDE losses for solving differential equations.
    """

    def __init__(self, config, dataset_name):
        super().__init__(config, dataset_name)
        assert self.config["derivative_method"] == "finite_diff", "Only finite difference method is supported"

        self.n_channels = 1
        self.pde_residual_func = get_pde_residual(dataset_name)

    def init(self, ground_truth, normalizer=None):
        self.device = ground_truth.device
        self.resolution = ground_truth.shape[-1]
        self.ground_truth = ground_truth

    def get_observation_loss(self, x_pred):
        """Calculate the PDE loss based on dataset type.

        Args:
            x_pred (torch.Tensor): Predicted data

        Returns:
            torch.Tensor: PDE loss
        """

        pde_residual = self.pde_residual_func(x_pred)
        n_obs = pde_residual.shape[-1] ** 2
        # mask = torch.where(torch.abs(pde_residual) > 1, 0, 1)
        # pde_residual = pde_residual * mask
        # n_obs = torch.sum(mask)
        loss = self.loss_func(pde_residual, n_obs)
        loss = loss.sum(dim=1, keepdim=True)
        return loss


def get_observation_class(config, dataset_name):
    """Get the observation class based on the configuration.

    Args:
        config (dict): Configuration dictionary for the observation
        dataset_name (str): Name of the dataset

    Returns:
        Observation: Observation class instance
    """
    if config["type"] == "full":
        return FullObservation(config, dataset_name)
    elif config["type"] == "sparse":
        return SparseObservation(config, dataset_name)
    elif config["type"] == "pde":
        return PDEObservation(config, dataset_name)
    else:
        raise ValueError(f"Unknown observation type: {config['type']}")
