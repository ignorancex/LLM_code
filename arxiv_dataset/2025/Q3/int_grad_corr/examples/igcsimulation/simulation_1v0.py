"""
Benchmark of IGC with pRF-like simulations.

This model is not a deep network with learnable parameters. It uses original
functions defining scrutinized image statistics.
"""

import os

import numpy as np
import torch
from igc import Gradients, IntegratedGradients, IntGradCorr
from igc.bsc import BaselineShapley, BslShapCorr
from igc.igac import IntGradAutoCorr
from igc.naive import IntGradMeanStd, NaiveCorrelation, NaiveTTest
from scipy.fft import next_fast_len
from scipy.stats import norm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from torchutils import AbstractModel, set_worker_seed

from .imgstat_1v0 import (
    argmax_sim_rand01_bin,
    max_mean_bin,
    max_sim_rand00_bin,
    w_sum,
)
from .mask_1v0 import Mask


IMGSTAT_INFO = {
    # Image statistic name: (extraction func., categorical, binary, n_y/n_cat)
    "w_sum": (w_sum, False, False, 1),
    "max_mean_bin": (max_mean_bin, False, True, 1),
    "max_sim_rand00_bin": (max_sim_rand00_bin, False, True, 1),
    "argmax_sim_rand01_bin": (argmax_sim_rand01_bin, True, True, 4),
}


# Dataset


class _Dataset(TorchDataset):
    def __init__(
        self, n_samples, mask, func, gain, imst_kwargs, y_std, seed=None
    ):
        self.n_samples = n_samples
        self.mask = mask.torch()
        self.img_size = mask.img_size
        self.func = func
        self.imst_kwargs = imst_kwargs
        self.y_std = y_std
        # Image generation
        self.gain = gain
        self.mean = torch.zeros(
            (self.img_size, self.img_size), dtype=torch.float32
        )
        self.std = torch.ones(
            (self.img_size, self.img_size), dtype=torch.float32
        )
        # Random generator
        self.rng = None
        if seed is not None:
            self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Image generation
        x_fft = torch.complex(
            torch.normal(self.mean, self.std, generator=self.rng),
            torch.normal(self.mean, self.std, generator=self.rng),
        )
        x_fft *= self.gain
        x = torch.fft.ifft2(x_fft, norm="ortho")  # pylint: disable=E1102
        x = x.real.to(torch.float32)
        # Compute image statistic
        y = self.func(
            x.unsqueeze(dim=0), self.mask, self.rng, **self.imst_kwargs
        )
        if self.y_std is not None:
            y -= self.y_std[0]
            y /= self.y_std[1]
        return x.squeeze(dim=0), y.squeeze(dim=0)


class Dataset:
    def __init__(
        self,
        mask_name,
        imst_name,
        img_size=64,
        n_samples=(100000, 10000),
        fft_slope=-1.2,
        imst_kwargs=None,
        normalize_imst=True,
        seed=0,
    ):
        self.img_size = img_size
        self.mask = Mask(mask_name, self.img_size)
        assert imst_name in IMGSTAT_INFO, "Unknown image statistic."
        self.imst_name = imst_name
        self.func, self.categorical, self.binary, self.n_y = IMGSTAT_INFO[
            self.imst_name
        ]
        if self.categorical:
            assert (
                self.mask.n_c > 1
            ), "Mask must have multiple channels for categorical statistics."
        if self.binary:
            assert self.mask.binary, "Mask is not binary."
        self.n_train, self.n_val = n_samples
        n_opt = next_fast_len(self.img_size)
        if n_opt != self.img_size:
            print(f"Optimal image size: {n_opt}")
        self.gain = self.compute_gain(fft_slope)
        self.imst_kwargs = imst_kwargs
        if self.imst_kwargs is None:
            self.imst_kwargs = {}
        self.seed = seed
        # Compute standardization values
        self.y_std = None
        if normalize_imst and (not self.categorical):
            self.compute_y_std_values()
        # Print info
        print(f"Train : {self.n_train}")
        print(f"Val   : {self.n_val}")

    @torch.no_grad()
    def compute_gain(self, fft_slope):
        freq = torch.fft.fftfreq(self.img_size)  # pylint: disable=E1102
        gain = torch.stack(torch.meshgrid(freq, freq, indexing="ij"), dim=0)
        gain = torch.linalg.norm(gain, dim=0)  # pylint: disable=E1102
        # Heuristic for zero frequency issues
        gain[0, 0] += 0.5 * gain[0, 1]
        # Apply slope in log-space
        gain = torch.pow(gain, fft_slope)
        # Scale gain, so that generated images are normalized
        gain /= torch.sqrt(torch.mean(gain**2))
        return gain

    @torch.no_grad()
    def compute_y_std_values(self):
        _, y = next(iter(self.val_dtld(batch_size=self.n_val)))
        self.y_std = (torch.mean(y).item(), torch.std(y).item())

    @torch.no_grad()
    def train_dtst(self, seed=100):
        return _Dataset(
            self.n_train,
            self.mask,
            self.func,
            self.gain,
            self.imst_kwargs,
            self.y_std,
            seed,
        )

    @torch.no_grad()
    def train_dtld(self, batch_size, seed=100, num_workers=0):
        # Indices provided to the dataset are not used by the generative
        # process. Only dataset's seed is important for reproducibility.
        return DataLoader(
            self.train_dtst(seed),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=set_worker_seed,
        )

    @torch.no_grad()
    def val_dtst(self, seed=None):
        if seed is None:
            seed = self.seed
        return _Dataset(
            self.n_val,
            self.mask,
            self.func,
            self.gain,
            self.imst_kwargs,
            self.y_std,
            seed,
        )

    # pylint: disable=W0613
    @torch.no_grad()
    def val_dtld(self, batch_size, seed=None, num_workers=0):
        # To guarantee reproducible validation samples, num_workers is disabled.
        return DataLoader(
            self.val_dtst(seed), batch_size=batch_size, pin_memory=True
        )


# Modules


class Encoder(nn.Module):
    def __init__(self, mask, func, y_std, categorical):
        super().__init__()
        self.register_buffer("mask", mask.torch(), persistent=False)
        self.func = func
        self.y_std = y_std
        self.kwargs = {}
        if categorical:
            self.kwargs = {"beta": 100.0}  # harder softmax

    def forward(self, x):
        y = self.func(x, self.mask, rng=None, **self.kwargs)
        if self.y_std is not None:
            y -= self.y_std[0]
            y /= self.y_std[1]
        return y


# Model


class Model(AbstractModel):
    # Default parameters
    seed = 100
    dtype = torch.float32
    project_path = os.path.dirname(__file__)

    def __init__(self, dataset, model_name="sim_1v0", device=None):
        super().__init__(model_name, device)
        # Dataset
        self.dtst = dataset
        # Network
        self.network = Encoder(
            self.dtst.mask,
            self.dtst.func,
            self.dtst.y_std,
            self.dtst.categorical,
        )
        self.network.to(self.device)

    def _get_dtld_kwargs(self, num_workers):
        return {"num_workers": num_workers, "pin_memory": True}

    def grad(
        self, n_x=None, y_idx=None, batch_size=None, x_seed=100, num_workers=0
    ):
        attr = Gradients(
            self.network,
            dataset=self.dtst.val_dtst(),
            dtld_kwargs=self._get_dtld_kwargs(num_workers),
        )
        return attr.compute(n_x, y_idx, batch_size, x_seed)

    def int_grad(
        self,
        n_x=None,
        x_0=None,
        y_idx=None,
        n_steps=64,
        batch_size=None,
        x_seed=100,
        x_0_seed=101,
        check_error=True,
        num_workers=0,
    ):
        attr = IntegratedGradients(
            self.network,
            dataset=self.dtst.val_dtst(),
            dtld_kwargs=self._get_dtld_kwargs(num_workers),
        )
        return attr.compute(
            n_x, x_0, y_idx, n_steps, batch_size, x_seed, x_0_seed, check_error
        )

    def int_grad_corr(
        self,
        x_0=None,
        y_idx=None,
        n_steps=64,
        batch_size=None,
        x_seed=100,
        x_0_seed=101,
        n_x=None,
        save_results=True,
        check_error=True,
        suffix="",
        num_workers=0,
    ):
        attr = IntGradCorr(
            self.network,
            dataset=self.dtst.val_dtst(),
            dtld_kwargs=self._get_dtld_kwargs(num_workers),
        )
        igc = attr.compute(
            x_0, y_idx, n_steps, batch_size, x_seed, x_0_seed, n_x, check_error
        )
        # Save results
        if suffix:
            suffix = "_" + suffix
        if save_results:
            np.savez(
                self.get_result_path(f"int_grad_corr{suffix}.npz"), data=igc
            )
        # Return results
        return igc

    def int_grad_auto_corr(
        self,
        x_0=None,
        y_idx=None,
        n_steps=64,
        batch_size=None,
        x_seed=100,
        x_0_seed=101,
        n_x=None,
        save_results=True,
        check_error=True,
        suffix="",
        num_workers=0,
    ):
        attr = IntGradAutoCorr(
            self.network,
            dataset=self.dtst.val_dtst(),
            dtld_kwargs=self._get_dtld_kwargs(num_workers),
        )
        igac = attr.compute(
            x_0, y_idx, n_steps, batch_size, x_seed, x_0_seed, n_x, check_error
        )
        # Save results
        if suffix:
            suffix = "_" + suffix
        if save_results:
            np.savez(
                self.get_result_path(f"int_grad_auto{suffix}.npz"), data=igac
            )
        # Return results
        return igac

    def int_grad_mean_std(
        self,
        x_0=None,
        y_idx=None,
        n_steps=64,
        batch_size=None,
        x_seed=100,
        x_0_seed=101,
        n_x=None,
        save_results=True,
        check_error=True,
        suffix="",
        num_workers=0,
    ):
        attr = IntGradMeanStd(
            self.network,
            dataset=self.dtst.val_dtst(),
            dtld_kwargs=self._get_dtld_kwargs(num_workers),
        )
        igm, igs = attr.compute(
            x_0, y_idx, n_steps, batch_size, x_seed, x_0_seed, n_x, check_error
        )
        # Save results
        if suffix:
            suffix = "_" + suffix
        if save_results:
            np.savez(
                self.get_result_path(f"int_grad_mean{suffix}.npz"), data=igm
            )
            np.savez(
                self.get_result_path(f"int_grad_std{suffix}.npz"), data=igs
            )
        # Return results
        return igm, igs

    def naive_corr(
        self,
        y_idx=None,
        batch_size=None,
        x_seed=100,
        n_x=None,
        save_results=True,
        suffix="",
        num_workers=0,
    ):
        attr = NaiveCorrelation(
            self.network,
            dataset=self.dtst.val_dtst(),
            dtld_kwargs=self._get_dtld_kwargs(num_workers),
        )
        corr = attr.compute(y_idx, batch_size, x_seed, n_x)
        # Save results
        if suffix:
            suffix = "_" + suffix
        if save_results:
            np.savez(self.get_result_path(f"corr{suffix}.npz"), data=corr)
        # Return results
        return corr

    def naive_ttest(
        self,
        y_idx=None,
        batch_size=None,
        x_seed=100,
        n_x=None,
        save_results=True,
        suffix="",
        num_workers=0,
    ):
        cat_ranges = (norm.ppf(0.1), norm.ppf(0.9))
        if self.dtst.categorical:
            cat_ranges = (0.5, 0.5)
        attr = NaiveTTest(
            self.network,
            dataset=self.dtst.val_dtst(),
            dtld_kwargs=self._get_dtld_kwargs(num_workers),
        )
        ttest = attr.compute(cat_ranges, y_idx, batch_size, x_seed, n_x)
        # Save results
        if suffix:
            suffix = "_" + suffix
        if save_results:
            np.savez(self.get_result_path(f"ttest{suffix}.npz"), data=ttest)
        # Return results
        return ttest

    @torch.no_grad()
    def igc_error(self, igc_name, y_idx=None, batch_size=None, num_workers=0):
        # Load IGC data
        igc = np.load(self.get_result_path(igc_name))["data"]
        # Compute IGC error
        attr = IntGradCorr(
            self.network,
            dataset=self.dtst.val_dtst(),
            dtld_kwargs=self._get_dtld_kwargs(num_workers),
        )
        error = attr.error(igc, y_idx, batch_size)
        # Return results
        return error

    @torch.no_grad()
    def igac_error(self, igac_name):
        # Load IGaC data
        igac = np.load(self.get_result_path(igac_name))["data"]
        # Compute IGaC error
        attr = IntGradAutoCorr(self.network, dataset=self.dtst.val_dtst())
        error = attr.error(igac)
        # Return results
        return error

    @torch.no_grad()
    def bsl_shap(
        self,
        x=None,
        x_0=None,
        y_idx=None,
        n_iter=8,
        x_0_batch_size=None,
        x_seed=100,
        x_0_seed=101,
        check_error=True,
        num_workers=0,
    ):
        attr = BaselineShapley(
            self.network,
            dataset=self.dtst.val_dtst(),
            dtld_kwargs=self._get_dtld_kwargs(num_workers),
        )
        return attr.compute(
            x,
            x_0,
            y_idx,
            n_iter,
            x_0_batch_size,
            x_seed,
            x_0_seed,
            check_error,
        )

    @torch.no_grad()
    def bsl_shap_corr(
        self,
        x_0=None,
        y_idx=None,
        n_iter=8,
        x_0_batch_size=None,
        x_seed=100,
        x_0_seed=101,
        n_x=None,
        save_results=True,
        check_error=True,
        suffix="",
        num_workers=0,
    ):
        attr = BslShapCorr(
            self.network,
            dataset=self.dtst.val_dtst(),
            dtld_kwargs=self._get_dtld_kwargs(num_workers),
        )
        bsc = attr.compute(
            x_0,
            y_idx,
            n_iter,
            x_0_batch_size,
            x_seed,
            x_0_seed,
            n_x,
            check_error,
        )
        # Save results
        if suffix:
            suffix = "_" + suffix
        if save_results:
            np.savez(
                self.get_result_path(f"bsl_shap_corr{suffix}.npz"), data=bsc
            )
        # Return results
        return bsc

    @torch.no_grad()
    def bsc_error(self, bsc_name, y_idx=None, batch_size=None, num_workers=0):
        # Load BSC data
        bsc = np.load(self.get_result_path(bsc_name))["data"]
        # Compute BSC error
        attr = BslShapCorr(
            self.network,
            dataset=self.dtst.val_dtst(),
            dtld_kwargs=self._get_dtld_kwargs(num_workers),
        )
        error = attr.error(bsc, y_idx, batch_size)
        # Return results
        return error
