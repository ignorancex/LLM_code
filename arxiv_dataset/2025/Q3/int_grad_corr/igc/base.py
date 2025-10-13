"""
Base class for all attribution methods.
"""

from abc import abstractmethod
from copy import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


# Data manager


def _greatest_divisor(a, b):
    c = min(a, b)
    for i in range(c):
        if not a % (c - i):
            return c - i
    return 1


class _SeedGenerator:
    def __init__(self, seed=None, n_digits=9):
        self.rng = np.random.default_rng(seed)
        self.high = 10**n_digits

    def get_seed(self):
        return int(self.rng.integers(self.high, size=1)[0])


class DataManager:
    """
    Help to setup the appropriate dataloaders for each context.

    Initialized dataloaders iterate over inputs :obj:`x`, *true* outputs
    :obj:`y`, baselines :obj:`x_0`, output component indices :obj:`y_idx`, or
    hidden layer component indices :obj:`z_idx`.

    Parameters
    ----------
    attr : AbstractAttributionMethod
        Attribution method.
    y_required : bool
        Define if *true* outputs :obj:`y` are required by :attr:`x_dtld`.

    Attributes
    ----------
    n_x : int
        Number of :obj:`x` samples.
    x_bsz : int
        Batch size of :attr:`x_dtld`.
    x_nb : int
        Number of batches of :attr:`x_dtld`.
    x_dtld : torch.utils.data.DataLoader | tuple(ArrayLike)
        Dataloader iterating over inputs :obj:`x` (and *true* outputs :obj:`y`
        if :obj:`y_required`).
    n_x_0 : int
        Number of :obj:`x_0` baselines.
    x_0_bsz : int
        Batch size of :attr:`x_0_dtld`.
    x_0_nb : int
        Number of batches of :attr:`x_0_dtld`.
    x_0_dtld : torch.utils.data.DataLoader | tuple(ArrayLike)
        Dataloader iterating over baselines :obj:`x_0`.
    n_z_idx : int
        Number of :obj:`z_idx` component indices.
    z_idx_bsz : int
        Batch size of :attr:`z_idx_dtld`.
    z_idx_nb : int
        Number of batches of :attr:`z_idx_dtld`.
    z_idx_dtld : torch.utils.data.DataLoader | tuple(ArrayLike)
        Dataloader iterating over hidden layer component indices :obj:`z_idx`.
    n_y_idx : int
        Number of :obj:`y_idx` component indices.
    y_idx_bsz : int
        Batch size of :attr:`y_idx_dtld`.
    y_idx_nb : int
        Number of batches of :attr:`y_idx_dtld`.
    y_idx_dtld : torch.utils.data.DataLoader | tuple(ArrayLike)
        Dataloader iterating over output component indices :obj:`y_idx`.
    """

    def __init__(self, attr, y_required=True):
        self.attr = attr
        self.y_required = y_required
        # Init x attributes
        self.n_x = None
        self.x_bsz = None
        self.x_nb = None
        self.x_dtld = None
        self.x_sampled = False
        # Init x_0 attributes
        self.n_x_0 = None
        self.x_0_bsz = None
        self.x_0_nb = None
        self.x_0_dtld = None
        self.x_0_seed_rng = None
        # Init z_idx attributes
        self.n_z_idx = getattr(self.attr, "z_size", None)
        self.z_idx_bsz = None
        self.z_idx_nb = None
        self.z_idx_dtld = None
        self.use_z_idx = getattr(self.attr, "use_z", False)
        # Init y_idx attributes
        self.n_y_idx = None
        self.y_idx_bsz = None
        self.y_idx_nb = None
        self.y_idx_dtld = None

    def get_x_dtype(self, numpy=False):
        """
        Return the PyTorch data types of all inputs :obj:`x`.

        Parameters
        ----------
        numpy : bool
            If `True`, it returns NumPy data types instead.

        Returns
        -------
        tuple(dtype)
            Data types of all inputs :obj:`x`.
        """
        if numpy:
            x_dtype = (self.attr.dtype_np,) * (
                len(self.attr.x_size) - self.attr.embedding_n_cat
            )
            x_dtype += (self.attr.dtype_cat_np,) * self.attr.embedding_n_cat
            return x_dtype
        x_dtype = (self.attr.dtype,) * (
            len(self.attr.x_size) - self.attr.embedding_n_cat
        )
        x_dtype += (self.attr.dtype_cat,) * self.attr.embedding_n_cat
        return x_dtype

    @torch.no_grad()
    def _check_x(self, x):
        # Check if 'y' is not required
        if x is None or isinstance(x, int):
            self.x_sampled = True
        else:
            assert not self.y_required, (
                "The attribution method requires 'y', "
                "so 'x' must be sampled from a dataset."
            )
        # x is sampled from the whole dataset
        if x is None:
            self.n_x = len(self.attr.dataset)
        # x is sampled n_x times from the dataset
        elif isinstance(x, int):
            assert x > 0, "The number of 'x' samples must be positive."
            self.n_x = x
            x = None
        # Predefined x
        else:
            # Multi x
            if not self.attr.multi_x:
                x = (x,)
            # Make x a tensor on the device
            x = tuple(
                torch.as_tensor(x_i, dtype=x_dtype_i, device=self.attr.device)
                for x_i, x_dtype_i in zip(x, self.get_x_dtype())
            )
            # Expand batch dimension if necessary
            x = tuple(
                x_i.unsqueeze(dim=0) if x_i.dim() == len(x_size_i) else x_i
                for x_i, x_size_i in zip(x, self.attr.x_size)
            )
            # Check x sizes
            assert len(x) == len(
                self.attr.x_size
            ), f"The number of 'x' inputs must be {len(self.attr.x_size)}"
            for i, (x_i, x_size_i) in enumerate(zip(x, self.attr.x_size)):
                assert (
                    x_i.size()[1:] == x_size_i
                ), f"'x' input number {i} should have shape {x_size_i}."
            # Get n_x
            self.n_x = x[0].size(0)
            # Multi x
            if not self.attr.multi_x:
                x = x[0]
        return x

    @torch.no_grad()
    def _check_x_0(self, x_0):
        # Baseline is set to a zero tensor
        if x_0 is None:
            x_0 = 0.0
        # Baselines are sampled n_x_0 times from the dataset
        if isinstance(x_0, int):
            assert x_0 > 0, "The number of baselines must be positive."
            assert x_0 <= len(self.attr.dataset), (
                f"The number of baselines ({x_0}) is larger than the number "
                f"of dataset samples ({len(self.attr.dataset)})."
            )
            self.n_x_0 = x_0
            x_0 = None
        # Zero and uniform baselines
        elif isinstance(x_0, float):
            self.n_x_0 = 1
            x_0 = tuple(
                torch.full(
                    (1,) + sz_i, x_0, dtype=dt_i, device=self.attr.device
                )
                for sz_i, dt_i in zip(self.attr.x_size, self.get_x_dtype())
            )
        # Predefined baselines
        else:
            # Multi x
            if not self.attr.multi_x:
                x_0 = (x_0,)
            # Make x a tensor on the device
            x_0 = tuple(
                torch.as_tensor(x_0_i, dtype=x_dtype_i, device=self.attr.device)
                for x_0_i, x_dtype_i in zip(x_0, self.get_x_dtype())
            )
            # Expand batch dimension if necessary
            x_0 = tuple(
                x_0_i.unsqueeze(dim=0) if x_0_i.dim() == len(sz_i) else x_0_i
                for x_0_i, sz_i in zip(x_0, self.attr.x_size)
            )
            # Check x_0 sizes
            assert len(x_0) == len(
                self.attr.x_size
            ), f"The number of 'x_0' inputs must be {len(self.attr.x_size)}"
            for i, (x_0_i, x_size_i) in enumerate(zip(x_0, self.attr.x_size)):
                assert (
                    x_0_i.size()[1:] == x_size_i
                ), f"'x_0' input number {i} should have shape {x_size_i}."
            # Get n_x_0
            self.n_x_0 = x_0[0].size(0)
            # Multi x
            if not self.attr.multi_x:
                x_0 = x_0[0]
        return x_0

    @torch.no_grad()
    def _check_y_idx(self, y_idx):
        # Select all components
        if y_idx is None:
            self.n_y_idx = self.attr.y_size
            y_idx = torch.arange(
                self.attr.y_size, dtype=torch.int64, device=self.attr.device
            )
        # Select one specific component
        elif isinstance(y_idx, int):
            self.n_y_idx = 1
            y_idx = torch.full(
                (1,), y_idx, dtype=torch.int64, device=self.attr.device
            )
        # Multiple components
        else:
            self.n_y_idx = len(y_idx)
            y_idx = torch.as_tensor(
                y_idx, dtype=torch.int64, device=self.attr.device
            )
        # Check y_idx
        assert (
            torch.max(y_idx).item() < self.attr.y_size
        ), f"Maximum 'y_idx' is {self.attr.y_size - 1}."
        return y_idx

    def _check_batchsizes(self, batch_size, use_x_0=True, use_y_idx=True):
        assert self.n_x is not None, "Check 'x' first."
        if use_x_0:
            assert self.n_x_0 is not None, "Check 'x_0' first."
        if use_y_idx:
            assert self.n_y_idx is not None, "Check 'y_idx' first."
        # Expand tuple of batch sizes
        if (
            isinstance(batch_size, tuple)
            and use_x_0
            and use_y_idx
            and self.use_z_idx
        ):
            self.x_bsz, self.x_0_bsz, self.z_idx_bsz = batch_size
            batch_size = None
        elif isinstance(batch_size, tuple) and use_x_0 and use_y_idx:
            self.x_bsz, self.x_0_bsz, self.y_idx_bsz = batch_size
            batch_size = None
        elif isinstance(batch_size, tuple) and use_y_idx:
            self.x_bsz, self.y_idx_bsz = batch_size
            batch_size = None
        elif isinstance(batch_size, tuple):
            self.x_bsz = batch_size
            batch_size = None
        # Fill batch sizes set to None:
        if (batch_size is None) and (self.x_bsz is None):
            self.x_bsz = 1
        elif self.x_bsz is None:
            self.x_bsz = self.n_x
        if use_x_0 and (self.x_0_bsz is None):
            self.x_0_bsz = self.n_x_0
        if use_y_idx and self.use_z_idx and (self.z_idx_bsz is None):
            self.z_idx_bsz = self.n_z_idx
        elif use_y_idx and (self.y_idx_bsz is None):
            self.y_idx_bsz = self.n_y_idx
        # Compute total batchsize if None
        if batch_size is None:
            batch_size = self.x_bsz
            if use_x_0:
                batch_size *= self.x_0_bsz
            if use_y_idx and self.use_z_idx:
                batch_size *= self.z_idx_bsz
            elif use_y_idx:
                batch_size *= self.y_idx_bsz
        # Adjust batch sizes
        batch_size = max(1, batch_size)
        if use_y_idx and self.use_z_idx:
            self.y_idx_bsz = self.n_y_idx
            # Compute z_bsz and z_nb
            self.z_idx_bsz = min(self.n_z_idx, self.z_idx_bsz, batch_size)
            self.z_idx_nb = int(np.ceil(self.n_z_idx / self.z_idx_bsz))
            batch_size = max(1, batch_size // self.z_idx_bsz)
        elif use_y_idx:
            # Compute y_idx_bsz and y_idx_nb
            self.y_idx_bsz = min(self.n_y_idx, self.y_idx_bsz, batch_size)
            self.y_idx_nb = int(np.ceil(self.n_y_idx / self.y_idx_bsz))
            batch_size = max(1, batch_size // self.y_idx_bsz)
        # Compute x_0_bsz and x_0_nb
        if use_x_0:
            self.x_0_bsz = _greatest_divisor(
                self.n_x_0, min(self.x_0_bsz, batch_size)
            )
            self.x_0_nb = self.n_x_0 // self.x_0_bsz
            batch_size = max(1, batch_size // self.x_0_bsz)
        # Compute x_bsz and x_nb
        if self.x_sampled:
            if use_x_0:
                self.x_bsz = min(
                    self.n_x // self.x_0_bsz, self.x_bsz, batch_size
                )
            else:
                self.x_bsz = min(self.n_x, self.x_bsz, batch_size)
            n_x_dropped = self.n_x % self.x_bsz
            if n_x_dropped:
                print(f"{n_x_dropped} 'x' samples dropped for efficiency.")
            self.x_nb = self.n_x // self.x_bsz
            self.n_x = self.x_bsz * self.x_nb
        else:
            self.x_bsz = _greatest_divisor(
                self.n_x, min(self.x_bsz, batch_size)
            )
            self.x_nb = self.n_x // self.x_bsz
        # Print actual batchsizes
        if use_x_0 and use_y_idx and self.use_z_idx:
            batch_size = self.x_bsz * self.x_0_bsz * self.z_idx_bsz
            print(
                f"batch size: {batch_size} ({self.x_bsz}, {self.x_0_bsz}, "
                f"{self.z_idx_bsz})"
            )
        elif use_x_0 and use_y_idx:
            batch_size = self.x_bsz * self.x_0_bsz * self.y_idx_bsz
            print(
                f"batch size: {batch_size} ({self.x_bsz}, {self.x_0_bsz}, "
                f"{self.y_idx_bsz})"
            )
        elif use_y_idx:
            batch_size = self.x_bsz * self.y_idx_bsz
            print(f"batch size: {batch_size} ({self.x_bsz}, {self.y_idx_bsz})")
        else:
            print(f"batch size: {self.x_bsz}")
        return self

    def _get_dataset_copy(self, seed):
        dataset = copy(self.attr.dataset)
        if dataset.rng is not None:
            dataset.rng = torch.Generator().manual_seed(seed)
        return dataset

    @torch.no_grad()
    def _init_x_dtld(self, x, x_seed):
        assert self.x_bsz is not None, "Check batch sizes first."
        # x sampled from the dataset
        if x is None:
            shuffle, rng = False, None
            if x_seed is not None:
                shuffle = True
                rng = torch.Generator().manual_seed(x_seed)
            self.x_dtld = DataLoader(
                self._get_dataset_copy(x_seed),
                self.x_bsz,
                shuffle,
                generator=rng,
                **self.attr.dtld_kwargs,
            )
            return self
        # Predefined x
        if self.attr.multi_x:
            self.x_dtld = tuple(
                (
                    tuple(
                        x_i[i * self.x_bsz : (i + 1) * self.x_bsz] for x_i in x
                    ),
                    None,
                )
                for i in range(self.x_nb)
            )
        else:
            self.x_dtld = tuple(
                (x[i * self.x_bsz : (i + 1) * self.x_bsz], None)
                for i in range(self.x_nb)
            )
        return self

    @torch.no_grad()
    def _init_x_0_dtld(self, x_0, x_0_seed):
        assert self.x_0_bsz is not None, "Check batch sizes first."
        # Random baselines
        if x_0 is None:
            self.x_0_dtld = DataLoader(
                self._get_dataset_copy(x_0_seed),
                self.x_0_bsz * self.x_bsz,
                shuffle=True,
                generator=torch.Generator().manual_seed(x_0_seed),
                **self.attr.dtld_kwargs,
            )
            self.x_0_seed_rng = _SeedGenerator(x_0_seed)
            return self
        # Predefined baselines
        #   Repeat along batch dimension
        if self.attr.multi_x:
            x_0 = tuple(
                x_0_i.repeat_interleave(self.x_bsz, dim=0) for x_0_i in x_0
            )
        else:
            x_0 = x_0.repeat_interleave(self.x_bsz, dim=0)
        #   Build x_0_dtld
        bsz = self.x_bsz * self.x_0_bsz
        if self.attr.multi_x:
            self.x_0_dtld = tuple(
                (tuple(x_0_i[i * bsz : (i + 1) * bsz] for x_0_i in x_0), None)
                for i in range(self.x_0_nb)
            )
        else:
            self.x_0_dtld = tuple(
                (x_0[i * bsz : (i + 1) * bsz], None) for i in range(self.x_0_nb)
            )
        return self

    @torch.no_grad()
    def _init_z_idx_dtld(self, n_steps=1, repeat=True):
        x_x_0_bsz = 1
        if (repeat is True) and (self.x_0_bsz is not None):
            x_x_0_bsz = self.x_bsz * self.x_0_bsz
        elif (repeat is True) or (repeat == "bsc"):
            x_x_0_bsz = self.x_bsz
        # Fill z_idx with zeros to ensure even batchsizes
        n_fill = self.z_idx_nb * self.z_idx_bsz - self.n_z_idx
        z_idx = torch.concat(
            (
                torch.arange(
                    self.n_z_idx, dtype=torch.int64, device=self.attr.device
                ),
                torch.zeros(n_fill, dtype=torch.int64, device=self.attr.device),
            )
        )
        # Repeat along batch dimension
        z_idx = z_idx.repeat_interleave(x_x_0_bsz)
        # Build z_idx_dtld
        bsz = x_x_0_bsz * self.z_idx_bsz
        self.z_idx_dtld = tuple(
            z_idx[i * bsz : (i + 1) * bsz].repeat(n_steps)
            for i in range(self.z_idx_nb)
        )
        return self

    @torch.no_grad()
    def _init_y_idx_dtld(self, y_idx, n_steps=1, repeat=True):
        assert n_steps > 0, "The number of steps must be positive."
        assert self.y_idx_bsz is not None, "Check batch sizes first."
        if self.use_z_idx:
            self._init_z_idx_dtld(n_steps, repeat)
            y_idx_ = y_idx.cpu().numpy()
            self.attr.final_lin_weight = self.attr.final_lin_weight[y_idx_]
            if self.attr.final_lin_bias is not None:
                self.attr.final_lin_bias = self.attr.final_lin_bias[y_idx_]
            return self
        x_x_0_bsz = 1
        if (repeat is True) and (self.x_0_bsz is not None):
            x_x_0_bsz = self.x_bsz * self.x_0_bsz
        elif (repeat is True) or (repeat == "bsc"):
            x_x_0_bsz = self.x_bsz
        # Fill y_idx with zeros to ensure even batchsizes
        n_fill = self.y_idx_nb * self.y_idx_bsz - self.n_y_idx
        y_idx_ = torch.concat(
            (
                y_idx,
                torch.zeros(n_fill, dtype=torch.int64, device=self.attr.device),
            )
        )
        # Repeat along batch dimension
        y_idx_ = y_idx_.repeat_interleave(x_x_0_bsz)
        # Build y_idx_dtld
        bsz = x_x_0_bsz * self.y_idx_bsz
        self.y_idx_dtld = tuple(
            y_idx_[i * bsz : (i + 1) * bsz].repeat(n_steps)
            for i in range(self.y_idx_nb)
        )
        return self

    def update_x_0_dtld_seed(self):
        """
        Update the seed of baseline dataloader after its initialization.

        If the dataset associated with the dataloader has an attribute
        :obj:`dtld.dataset.rng`, that represents a random number generator
        (:obj:`torch.Generator`), it will be updated with the same seed.

        Returns
        -------
        self
        """
        if self.x_0_seed_rng is not None:
            seed = self.x_0_seed_rng.get_seed()
            if self.x_0_dtld.generator is None:
                self.x_0_dtld.generator = torch.Generator()
            self.x_0_dtld.generator.manual_seed(seed)
            rng = getattr(self.x_0_dtld.dataset, "rng", None)
            if rng is not None and isinstance(rng, torch.Generator):
                rng.manual_seed(seed)
        return self

    def add_data(self, x, x_0, y_idx, n_steps, batch_size, x_seed, x_0_seed):
        """
        Setup a data manager iterating over inputs :obj:`x`, baselines
        :obj:`x_0`, and output component indices :obj:`y_idx` (or :obj:`z_idx`).

        Parameters
        ----------
        x : None | int | ArrayLike | tuple(ArrayLike)
            - None : :attr:`x_dtld` iterates over the whole dataset.
            - int : Number of :obj:`x` inputs sampled from the dataset.
            - ArrayLike | tuple(ArrayLike) : Set new :obj:`x` used by :attr:`x_dtld`.
        x_0 : None | int | float | ArrayLike | tuple(ArrayLike)
            - None : Zero baseline :obj:`x_0`.
            - int : Number of :obj:`x_0` baselines sampled from the dataset.
            - float : Constant baseline :obj:`x_0`.
            - ArrayLike | tuple(ArrayLike) : Set :obj:`x_0` baselines used by :attr:`x_0_dtld`.
        y_idx : None | int | ArrayLike
            - None : :attr:`y_idx_dtld` iterates over all output component indices :obj:`y_idx`.
            - int : Select a specific output component index :obj:`y_idx`.
            - ArrayLike : Select multiple output component indices :obj:`y_idx`.
        n_steps : int
            Number of steps of the Riemann approximation of supporting
            Integrated Gradients (IG) (see
            :cite:`SundararajanAxiomaticAttributionDeep2017` for details).
        batch_size : None | int | tuple(int)
            - None : Set :attr:`x_bsz` = 1, :attr:`x_0_bsz` = :attr:`n_x_0`, and :attr:`y_idx_bsz` = :attr:`n_y_idx` (or :attr:`z_idx_bsz` = :attr:`n_z_idx`).
            - int : Total batch size budget automatically distributed between :attr:`x_bsz`, :attr:`x_0_bsz`, and :attr:`y_idx_bsz` (or :attr:`z_idx_bsz`).
            - tuple(int) : Set :attr:`x_bsz`, :attr:`x_0_bsz`, and :attr:`y_idx_bsz` (or :attr:`z_idx_bsz`) individually.
        x_seed : None | int
            Seed associated with :attr:`x_dtld`.
        x_0_seed : None | int
            Seed associated with :attr:`x_0_dtld`.

        Returns
        -------
        torch.Tensor
            Resolved :obj:`y_idx` if it was :obj:`None`.
        """
        x = self._check_x(x)
        x_0 = self._check_x_0(x_0)
        y_idx = self._check_y_idx(y_idx)
        self._check_batchsizes(batch_size)
        self._init_x_dtld(x, x_seed)
        self._init_x_0_dtld(x_0, x_0_seed)
        self._init_y_idx_dtld(y_idx, n_steps)
        return y_idx

    def add_data_iter_x_y_idx(self, x, y_idx, batch_size, x_seed):
        """
        Setup a data manager iterating over inputs :obj:`x` and output component
        indices :obj:`y_idx`.

        Parameters
        ----------
        x : None | int | ArrayLike | tuple(ArrayLike)
            - None : :attr:`x_dtld` iterates over the whole dataset.
            - int : Number of :obj:`x` inputs sampled from the dataset.
            - ArrayLike | tuple(ArrayLike) : Set new :obj:`x` used by :attr:`x_dtld`.
        y_idx : None | int | ArrayLike
            - None : :attr:`y_idx_dtld` iterates over all output component indices :obj:`y_idx`.
            - int : Select a specific output component index :obj:`y_idx`.
            - ArrayLike : Select multiple output component indices :obj:`y_idx`.
        batch_size : None | int | tuple(int)
            - None : Set :attr:`x_bsz` = 1 and :attr:`y_idx_bsz` = :attr:`n_y_idx`.
            - int : Total batch size budget automatically distributed between :attr:`x_bsz` and :attr:`y_idx_bsz`.
            - tuple(int) : Set :attr:`x_bsz` and :attr:`y_idx_bsz` individually.
        x_seed : None | int
            Seed associated with :attr:`x_dtld`.

        Returns
        -------
        torch.Tensor
            Resolved :obj:`y_idx` if it was :obj:`None`.
        """
        assert not self.use_z_idx, (
            "Incompatible with the use of hidden layer component "
            "indices :obj:`z_idx`."
        )
        x = self._check_x(x)
        y_idx = self._check_y_idx(y_idx)
        self._check_batchsizes(batch_size, use_x_0=False)
        self._init_x_dtld(x, x_seed)
        self._init_y_idx_dtld(y_idx)
        return y_idx

    def add_data_iter_x(self, x, y_idx, batch_size, x_seed):
        """
        Setup a data manager iterating over inputs :obj:`x`.

        Parameters
        ----------
        x : None | int | ArrayLike | tuple(ArrayLike)
            - None : :attr:`x_dtld` iterates over the whole dataset.
            - int : Number of :obj:`x` inputs sampled from the dataset.
            - ArrayLike | tuple(ArrayLike) : Set new :obj:`x` used by :attr:`x_dtld`.
        y_idx : None | int | ArrayLike
            Selected output component indices. If :obj:`None`, :obj:`y_idx` is
            resolved to all output component indices.
        batch_size : None | int
            - None : Set :attr:`x_bsz` = 1.
            - int : Set :attr:`x_bsz`.
        x_seed : None | int
            Seed associated with :attr:`x_dtld`.

        Returns
        -------
        torch.Tensor
            Resolved :obj:`y_idx` if it was :obj:`None`.
        """
        assert (batch_size is None) or isinstance(
            batch_size, int
        ), ":obj:`batch_size` must be an integer."
        x = self._check_x(x)
        y_idx = self._check_y_idx(y_idx)
        self._check_batchsizes(batch_size, use_x_0=False, use_y_idx=False)
        self._init_x_dtld(x, x_seed)
        return y_idx

    def add_data_naive(self, x, y_idx, batch_size, x_seed):
        """
        Setup a data manager dedicated to naive attribution methods (
        :class:`igc.naive.NaiveCorrelation` and :class:`igc.naive.NaiveTTest`).

        Parameters
        ----------
        x : None | int | ArrayLike | tuple(ArrayLike)
            - None : :attr:`x_dtld` iterates over the whole dataset.
            - int : Number of :obj:`x` inputs sampled from the dataset.
            - ArrayLike | tuple(ArrayLike) : Set new :obj:`x` used by :attr:`x_dtld`.
        y_idx : None | int | ArrayLike
            - None : :attr:`y_idx_dtld` iterates over all output component indices :obj:`y_idx`.
            - int : Select a specific output component index :obj:`y_idx`.
            - ArrayLike : Select multiple output component indices :obj:`y_idx`.
        batch_size : None | int | tuple(int)
            - None : Set :attr:`x_bsz` = 1 and :attr:`y_idx_bsz` = :attr:`n_y_idx`.
            - int : Total batch size budget automatically distributed between :attr:`x_bsz` and :attr:`y_idx_bsz`.
            - tuple(int) : Set :attr:`x_bsz` and :attr:`y_idx_bsz` individually.
        x_seed : None | int
            Seed associated with :attr:`x_dtld`.

        Returns
        -------
        torch.Tensor
            Resolved :obj:`y_idx` if it was :obj:`None`.
        """
        assert not self.use_z_idx, (
            "Incompatible with the use of hidden layer component "
            "indices :obj:`z_idx`."
        )
        x = self._check_x(x)
        y_idx = self._check_y_idx(y_idx)
        self._check_batchsizes(batch_size, use_x_0=False)
        self._init_x_dtld(x, x_seed)
        self._init_y_idx_dtld(y_idx, repeat=False)
        return y_idx

    def add_data_bsc(
        self, x, x_0, y_idx, n_iter, x_0_batch_size, x_seed, x_0_seed
    ):
        """
        Setup a data manager dedicated to Baseline Shapley and Baseline Shapley
        Correlation attribution methods (:class:`igc.bsc.BaselineShapley` and
        :class:`igc.bsc.BslShapCorr`).

        Parameters
        ----------
        x : None | int | ArrayLike
            - None : :attr:`x_dtld` iterates over the whole dataset.
            - int : Number of :obj:`x` inputs sampled from the dataset.
            - ArrayLike : Set new :obj:`x` used by :attr:`x_dtld`.
        x_0 : None | int | float | ArrayLike
            - None : Zero baseline :obj:`x_0`.
            - int : Number of :obj:`x_0` baselines sampled from the dataset.
            - float : Constant baseline :obj:`x_0`.
            - ArrayLike : Set :obj:`x_0` baselines used by :attr:`x_0_dtld`.
        y_idx : None | int | ArrayLike
            - None : :attr:`y_idx_dtld` iterates over all output component indices :obj:`y_idx`.
            - int : Select a specific output component index :obj:`y_idx`.
            - ArrayLike : Select multiple output component indices :obj:`y_idx`.
        n_iter : int
            Number of iterations, i.e. the number of random sequences of input
            component indices enabled one after the other.
        x_0_batch_size : None | int
            - None : Set :attr:`x_0_bsz` = :attr:`n_x_0`.
            - int : Set :attr:`x_0_bsz`.
        x_seed : None | int
            Seed associated with :attr:`x_dtld`.
        x_0_seed : None | int
            Seed associated with :attr:`x_0_dtld`.

        Returns
        -------
        torch.Tensor
            Resolved :obj:`y_idx` if it was :obj:`None`.
        """
        assert not self.use_z_idx, (
            "Incompatible with the use of hidden layer component "
            "indices :obj:`z_idx`."
        )
        assert (x_0_batch_size is None) or isinstance(
            x_0_batch_size, int
        ), ":attr:`x_0_bsz` must be an integer."
        x = self._check_x(x)
        x_0 = self._check_x_0(x_0)
        y_idx = self._check_y_idx(y_idx)
        batch_size = (1, x_0_batch_size, 1)
        self._check_batchsizes(batch_size)
        self._init_x_dtld(x, x_seed)
        self._init_x_0_dtld(x_0, x_0_seed)
        self._init_y_idx_dtld(y_idx, n_iter * self.x_0_bsz, repeat="bsc")
        return y_idx


# Attributions base class


class AbstractAttributionMethod:
    """
    Define the base class for an abstract attribution method.

    The sub-classes are expected to implement a :meth:`compute` method, specific
    to each attribution method.

    Parameters
    ----------
    module : torch.nn.Module
        PyTorch module defining the model under scrutiny.
    dataset : torch.utils.data.Dataset
        PyTorch dataset providing inputs/outputs for any given index. See
        `PyTorch documentation <https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_
        for more information. In addition, inputs must be organized in a
        specific manner, see warning below.
    dtld_kwargs : dict
        Additional keyword arguments to the dataloaders
        (:obj:`torch.utils.data.DataLoader`) constructed around the
        :attr:`dataset`, except: :obj:`dataset`, :obj:`batch_size`,
        :obj:`shuffle`, :obj:`sampler`, :obj:`batch_sampler`, and
        :obj:`generator`.
    forward_method_name : str
        Name of the forward method of the :attr:`module`. If :const:`None`,
        the default :obj:`forward` is used.
    forward_method_kwargs : dict
        Additional keyword arguments to the forward method of the
        :attr:`module`.
    dtype : torch.dtype
        Default data type of all intermediary tensors. It also defines the NumPy
        data type of the attribution results.
    dtype_cat : torch.dtype
        Default data type of the categorical input tensors.

    Notes
    -----

    .. warning::
        When computing attributions on models using multiple inputs, e.g., x_1,
        x_2, and x_cat, with x_cat a categorical input, the dataset must return
        all inputs packed in a tuple, such as: (x_1, x_2, x_cat), y. Note that
        categorical inputs must be placed at the end of the tuple.
    """

    def __init__(
        self,
        module,
        dataset,
        dtld_kwargs=None,
        forward_method_name=None,
        forward_method_kwargs=None,
        dtype=torch.float32,
        dtype_cat=torch.int32,
    ):
        self.module, self.device = self._check_module(module)
        self.dataset = self._check_dataset(dataset)
        self.dtld_kwargs = self._check_kwargs(dtld_kwargs)
        self.forward_func = self._check_forward_func(forward_method_name)
        self.forward_func_kwargs = self._check_kwargs(forward_method_kwargs)
        self.dtype, self.dtype_cat, self.dtype_np, self.dtype_cat_np = (
            self._check_dtype(dtype, dtype_cat)
        )
        self.x_size, self.y_size, self.multi_x = self._get_x_y_sizes()
        # Init embedding parameters
        self.embedding_func = None
        self.embedding_func_kwargs = {}
        self.embedding_size = self.x_size
        self.embedding_n_cat = 0

    def _check_module(self, module):
        # Check module
        assert isinstance(
            module, nn.Module
        ), "Module must be inherited from torch.nn.Module."
        # Set module to eval mode
        module.eval()
        # Get device
        device = next(module.parameters(), None)
        if device is None:
            device = next(module.buffers(), None)
        assert device is not None, "Module with no parameter or buffer."
        return module, device.device

    def _check_kwargs(self, kwargs):
        if kwargs is None:
            kwargs = {}
        return kwargs

    def _check_dataset(self, dataset):
        assert isinstance(
            dataset, torch.utils.data.Dataset
        ), "Dataset must be inherited from torch.utils.data.Dataset."
        return dataset

    def _check_forward_func(self, forward_method_name):
        if forward_method_name is None:
            forward_method_name = "forward"
        return getattr(self.module, forward_method_name)

    def _check_dtype(self, dtype, dtype_cat):
        # Check default dtype and dtype_cat for categorical inputs
        assert dtype in (torch.float16, torch.float32, torch.float64)
        assert dtype_cat in (torch.int16, torch.int32, torch.int64)
        # Define dtype and dtype_cat for NumPy arrays
        dtype_np = np.float64
        if dtype is not torch.float64:
            dtype_np = np.float32
        dtype_cat_np = np.int64
        if dtype_cat is not torch.int64:
            dtype_cat_np = np.int32
        return dtype, dtype_cat, dtype_np, dtype_cat_np

    def _check_x_size(self, x_size):
        if isinstance(x_size, int):
            x_size = (x_size,)
        if not isinstance(x_size[0], tuple):
            x_size = (x_size,)
        for x_size_i in x_size:
            assert isinstance(x_size_i, tuple), (
                "Each input of the dataloader tuple x must be at least "
                "2-dimensional (batch size, ...). If an input is a simple "
                "scalar, it must be unsqueezed on the second dimension."
            )
        return x_size

    def _check_y_size(self, y_size):
        assert isinstance(y_size, int), "'y_size' must be an integer."
        return y_size

    @torch.no_grad()
    def _get_x_y_sizes_from_dtst(self):
        x, y = self.dataset[0]
        if isinstance(x, (tuple, list)):
            x_size = tuple(x_i.size() for x_i in x)
        else:
            x_size = (x.size(),)
        # Check y_size
        assert y.dim() == 1, (
            "Dateset true outputs 'y' must have shape (n features,). If 'y' is "
            "a scalar, it must be unsqueezed."
        )
        y_size = y.size(0)
        return x_size, y_size

    def _get_x_y_sizes(self):
        x_size, y_size = self._get_x_y_sizes_from_dtst()
        # Check x_size
        x_size = self._check_x_size(x_size)
        # Check y_size
        y_size = self._check_y_size(y_size)
        # Define multi_x
        multi_x = len(x_size) > 1
        return x_size, y_size, multi_x

    @torch.no_grad()
    def _get_embedding_n_cat_from_dtst(self):
        x, _ = self.dataset[0]
        if self.multi_x:
            x_dtype = tuple(x_i.dtype for x_i in x)
        else:
            x_dtype = x.dtype
        n_cat = 0
        for i, x_dtype_i in enumerate(x_dtype):
            if x_dtype_i in (torch.int16, torch.int32, torch.int64):
                n_cat += 1
            else:
                assert i < (len(x_dtype) - n_cat), (
                    "Categorical inputs must be placed at the end of the tuple "
                    "of inputs."
                )
        return n_cat

    @torch.no_grad()
    def _get_embedding_size_from_dtst(self):
        x, _ = self.dataset[0]
        if self.multi_x:
            x = tuple(x_i.unsqueeze(dim=0).to(self.device) for x_i in x)
        else:
            x = (x.unsqueeze(dim=0).to(self.device),)
        x_emb = self._emb(x)
        if isinstance(x_emb, (tuple, list)):
            return tuple(x_emb_i.size()[1:] for x_emb_i in x_emb)
        return (x_emb.size()[1:],)

    def _check_embedding_n_cat(self, embedding_n_cat):
        assert embedding_n_cat <= len(self.x_size), "Invalid 'embedding_n_cat'."
        return embedding_n_cat

    def add_embedding_method(
        self,
        embedding_method_name,
        embedding_method_kwargs=None,
        embedding_n_cat=None,
    ):
        """
        Add an embedding method to preprocess categorical inputs.

        .. note::
            Adding an embedding method modifies the output shapes of
            attributions associated with categorical inputs.

        .. warning::
            The effect of this method must be excluded from the forward method
            defined by :attr:`forward_method_name` at initialization.

        Parameters
        ----------
        embedding_method_name : str
            Name of the embedding method of the :attr:`module`.
        embedding_method_kwargs : dict
            Additional keyword arguments to the embedding method of the
            :attr:`module`.
        embedding_n_cat : int
            Number of categorical inputs. If :const:`None`, this value is
            inferred from the input data types (:obj:`torch.int16`,
            :obj:`torch.int32`, :obj:`torch.int64`).

        Returns
        -------
        self
        """
        self.embedding_func = getattr(self.module, embedding_method_name)
        self.embedding_func_kwargs = self._check_kwargs(embedding_method_kwargs)
        # Check embedding_n_cat
        if embedding_n_cat is None:
            embedding_n_cat = self._get_embedding_n_cat_from_dtst()
        self.embedding_n_cat = self._check_embedding_n_cat(embedding_n_cat)
        # Check embedding_size
        embedding_size = self._get_embedding_size_from_dtst()
        self.embedding_size = self._check_x_size(embedding_size)
        return self

    def _init_output(self, size_prefix, size, dtype=None):
        if dtype is None:
            dtype = (self.dtype_np,) * len(size)
        return tuple(
            np.zeros(size_prefix + size_i, dtype=dtype_i)
            for size_i, dtype_i in zip(size, dtype)
        )

    @torch.no_grad()
    def _record_y(self, y, y_idx, x_batch_size):
        if y is None:
            return None
        y_idx_ = y_idx.cpu().unsqueeze(dim=0).expand(x_batch_size, -1)
        y_ = torch.gather(y.cpu(), dim=1, index=y_idx_)
        return y_.numpy().astype(self.dtype_np)

    @torch.no_grad()
    def _emb(self, x):
        if self.embedding_func is None:
            return x
        # Apply embedding
        x_emb = self.embedding_func(  # pylint: disable=E1102
            *x[-self.embedding_n_cat :], **self.embedding_func_kwargs
        )
        if self.embedding_n_cat == 1:
            x_emb = (x_emb,)
        return x[: -self.embedding_n_cat] + x_emb

    @torch.no_grad()
    def _fwd_no_grad(self, x):
        return self.forward_func(*x, **self.forward_func_kwargs)

    def _fwd(self, x):
        # Prepare inputs
        for x_i in x:
            x_i.requires_grad_(True)
        # Eval
        return self.forward_func(*x, **self.forward_func_kwargs)

    def _bwd(self, x, y_r, y_idx):
        # Reset gradients
        for x_i in x:
            if x_i.grad is not None:
                x_i.grad = None
        self.module.zero_grad(set_to_none=True)
        # Compute gradients
        y_r_ = torch.gather(y_r, dim=1, index=y_idx.unsqueeze(dim=1))
        y_r_.backward(gradient=torch.ones_like(y_r_), retain_graph=True)
        y_r_ = y_r_.squeeze(dim=1).detach().cpu().numpy()
        x_grad = tuple(x_i.grad.cpu().numpy() for x_i in x)
        return y_r_, x_grad

    @abstractmethod
    def compute(self):
        """
        Abstract method computing attributions.
        """
        return

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)
