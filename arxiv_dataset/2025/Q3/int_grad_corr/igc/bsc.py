"""
Baseline Shapley Correlation (BSC) utilities.

.. note::
    For demonstration purposes only

.. note::
    Welford's online algorithm :cite:`WelfordNoteMethodCalculating1962`
    is used for the computation of mean, standard deviation, and correlation
    statistics (see `wikipedia`_).

.. _wikipedia: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
"""

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .base import AbstractAttributionMethod, DataManager


# Baseline Shapley


class BaselineShapley(AbstractAttributionMethod):
    """
    Baseline Shapley (BS).

    See the original paper :cite:`SundararajanmanyShapleyvalues2020` for more
    information.

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
    """

    @torch.no_grad()
    def _fwd_bs(self, x, y_idx):  # pylint: disable=W0221
        # Set module to eval mode
        self.module.eval()
        # Eval
        y_r = self.forward_func(x, **self.forward_func_kwargs)
        y_r = torch.gather(y_r, dim=1, index=y_idx.unsqueeze(dim=1))
        return y_r.squeeze(dim=1).cpu().numpy()

    @torch.no_grad()
    def _bsl_shap_per_x_per_y_idx(self, dtmg, x, y_idx, n_iter, x_0_seed):
        # Update x_0_dtld seed
        dtmg.update_x_0_dtld_seed()
        # Define permutation seeds
        perm_seed_rng = np.random.default_rng(x_0_seed)
        # Get the total number of features
        n_features = x.numel() // dtmg.x_0_bsz // n_iter
        # Compute baseline Shapley
        y_0 = 0.0
        y_r = self._fwd_bs(x[0].unsqueeze(dim=0), y_idx[0].unsqueeze(dim=0))[0]
        bsl_shap = np.zeros(x.size()[1:], dtype=np.float32)
        # Iterate over baselines
        for i, (x_0_i, _) in enumerate(dtmg.x_0_dtld):
            # Break when x_0_nb is reached
            if i == dtmg.x_0_nb:
                break
            # Prepare baseline
            x_0_i = x_0_i.to(self.device)
            # Compute y_0_i
            y_0_i = self._fwd_bs(x_0_i, y_idx[: dtmg.x_0_bsz])
            y_0 += np.sum(y_0_i)
            # Prepare baseline
            x_0_i = x_0_i.repeat(*((n_iter,) + (1,) * (x.dim() - 1)))
            # Prepare permutations
            feature_mask = np.arange(n_features)[None, :].repeat(
                dtmg.x_0_bsz * n_iter, axis=0
            )
            # Iterate over features
            prev_eval = np.tile(y_0_i, n_iter)
            mask = torch.zeros_like(x)
            feature_permutation = torch.as_tensor(
                perm_seed_rng.permuted(feature_mask, axis=1), device=self.device
            )
            for j in range(n_features):
                mask_j = nn.functional.one_hot(  # pylint: disable=E1102
                    feature_permutation[:, j], num_classes=n_features
                )
                mask_j = mask_j.view(x.size())
                mask += mask_j
                x_j = torch.where(mask != 0, x, x_0_i)
                modified_eval = self._fwd_bs(x_j, y_idx)
                eval_diff = modified_eval - prev_eval
                bsl_shap += np.sum(
                    eval_diff[(...,) + (None,) * (x.dim() - 1)]
                    * mask_j.cpu().numpy(),
                    axis=0,
                )
                prev_eval = modified_eval
        # Average baselines (and iterations)
        y_0 /= dtmg.n_x_0
        bsl_shap /= dtmg.n_x_0 * n_iter
        return y_0, y_r, bsl_shap

    @torch.no_grad()
    def _bsl_shap_per_x(self, dtmg, x, n_iter, x_0_seed):
        # Init outputs
        y_0 = np.zeros((dtmg.n_y_idx,), dtype=self.dtype_np)
        y_r = np.zeros((dtmg.n_y_idx,), dtype=self.dtype_np)
        bsl_shap = np.zeros(
            (dtmg.n_y_idx,) + self.embedding_size[0], dtype=self.dtype_np
        )
        # Iterate over y_idx
        for i, y_idx_i in enumerate(dtmg.y_idx_dtld):
            y_0_i, y_r_i, bsl_shap_i = self._bsl_shap_per_x_per_y_idx(
                dtmg, x, y_idx_i, n_iter, x_0_seed
            )
            # Record y_0 and y_r
            y_0[i] += y_0_i
            y_r[i] += y_r_i
            # Record baseline Shapley
            bsl_shap[i] += bsl_shap_i
        return y_0, y_r, bsl_shap

    @torch.no_grad()
    def compute(  # pylint: disable=W0221
        self,
        x,
        x_0=None,
        y_idx=None,
        n_iter=8,
        x_0_batch_size=1,
        x_seed=None,
        x_0_seed=100,
        check_error=True,
    ):
        """
        Compute Baseline Shapley (BS).

        .. warning::

            Baseline Shapley (BS) does not support multiple inputs.

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
        check_error : bool
            If :obj:`True`, the mean absolute error of BS approximations is
            reported. For each input, baseline, and output component, the
            *completeness* property of BS states that the sum of input component
            attributions must be equal to the difference between the model
            predictions associated with the input and baseline under scrutiny.

        Returns
        -------
        tuple
            - ArrayLike : sampled inputs
            - ArrayLike : corresponding *true* outputs
            - ArrayLike : model predictions of sampled inputs
            - ArrayLike : model predictions of baselines
            - ArrayLike : BS attributions of shape (:attr:`n_x`, :attr:`n_y_idx`, * unbatched :obj:`x` shape).
        """
        # Check for multi_x
        assert (
            not self.multi_x
        ), "Baseline Shapley (BS) does not support multiple inputs."
        # Set module to eval mode
        self.module.eval()
        # Init data manager
        dtmg = DataManager(self, y_required=False)
        y_idx = dtmg.add_data_bsc(
            x, x_0, y_idx, n_iter, x_0_batch_size, x_seed, x_0_seed
        )
        # Init outputs
        x_np = np.zeros((dtmg.n_x,) + self.x_size[0], dtype=self.dtype_np)
        y_np = np.zeros((dtmg.n_x, dtmg.n_y_idx), dtype=self.dtype_np)
        y_0 = np.zeros((dtmg.n_x, dtmg.n_y_idx), dtype=self.dtype_np)
        y_r = np.zeros((dtmg.n_x, dtmg.n_y_idx), dtype=self.dtype_np)
        bsl_shap = np.zeros(
            (dtmg.n_x, dtmg.n_y_idx) + self.embedding_size[0],
            dtype=self.dtype_np,
        )
        # Iterate over x
        for i, (x_i, y_i) in enumerate(
            tqdm(dtmg.x_dtld, total=dtmg.x_nb, desc="bs")
        ):
            # Break when x_nb is reached
            if i == dtmg.x_nb:
                break
            # Record x, y
            x_i_np = x_i.cpu().numpy()
            y_i_np = self._record_y(y_i, y_idx, dtmg.x_bsz)
            x_np[i] += x_i_np[0]
            if y_i_np is not None:
                y_np[i] += y_i_np[0]
            # Prepare x
            # Send x to the device
            x_i = x_i.to(self.device)
            # Embed discrete inputs
            x_i = self._emb((x_i,))[0]
            # Repeat x along batch dimension
            x_i = x_i.repeat(
                *((x_0_batch_size * n_iter,) + (1,) * (x_i.dim() - 1))
            )
            # Compute baseline Shapley
            y_0_i, y_r_i, bs_i = self._bsl_shap_per_x(
                dtmg, x_i, n_iter, x_0_seed
            )
            # Record y_0 and y_r
            y_0[i] += y_0_i
            y_r[i] += y_r_i
            # Record baseline Shapley
            bsl_shap[i] += bs_i
        # Check error
        if check_error:
            bsl_shap_sum = np.sum(
                bsl_shap.reshape((dtmg.n_x, dtmg.n_y_idx, -1)), axis=2
            )
            print(f"bs err: {np.mean(np.abs(bsl_shap_sum - y_r + y_0)):>9.6f}")
        # Return results
        return x_np, y_np, y_0, y_r, bsl_shap


# Baseline Shapley Correlation


class BslShapCorr(BaselineShapley):
    """
    Baseline Shapley Correlation (BSC).

    See the original paper :cite:`LelievreIntegratedGradientCorrelation2024` for
    more information.

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
    """

    @torch.no_grad()
    def compute(  # pylint: disable=W0221,W0237
        self,
        x_0=None,
        y_idx=None,
        n_iter=8,
        x_0_batch_size=1,
        x_seed=None,
        x_0_seed=100,
        n_x=None,
        check_error=True,
    ):
        """
        Compute Baseline Shapley Correlation (BSC).

        .. warning::

            Baseline Shapley Correlation (BSC) does not support multiple inputs.

        Parameters
        ----------
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
        n_x : None | int
            - None : :attr:`x_dtld` iterates over the whole dataset.
            - int : Number of :obj:`x` inputs sampled from the dataset.
        check_error : bool
            If :obj:`True`, the mean absolute error of BS and BSC approximations
            is reported. For each input, baseline, and output component, the
            *Completeness* property of BS states that the sum of input component
            attributions must be equal to the difference between the model
            predictions associated with the input and baseline under scrutiny.
            For each output component, the *completeness* property of BSC states
            that the sum of input component attributions must be equal to the
            correlation between model predictions and *true* outputs.

        Returns
        -------
        ArrayLike
            BSC attributions of shape (:attr:`n_y_idx`, * unbatched :obj:`x`
            shape)
        """
        # Check for multi_x
        assert not self.multi_x, (
            "Baseline Shapley Correlation (BSC) does not support multiple "
            "inputs."
        )
        # Set module to eval mode
        self.module.eval()
        # Init data manager
        dtmg = DataManager(self)
        y_idx = dtmg.add_data_bsc(
            n_x, x_0, y_idx, n_iter, x_0_batch_size, x_seed, x_0_seed
        )
        # Init outputs
        bs_error = 0.0
        y_mean = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_std = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_r_mean = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_r_std = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        corr = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        bsc = np.zeros(
            (dtmg.n_y_idx,) + self.embedding_size[0], dtype=self.dtype_np
        )
        bsc_mean = np.zeros(
            (dtmg.n_y_idx,) + self.embedding_size[0], dtype=self.dtype_np
        )
        # Iterate over x
        postfix = None
        if check_error:
            postfix = "bs err: ?"
        tqdm_iterator = tqdm(
            dtmg.x_dtld, total=dtmg.x_nb, desc="bsc", postfix=postfix
        )
        for i, (x_i, y_i) in enumerate(tqdm_iterator):
            n_x_count = (i + 1) * dtmg.x_bsz
            # Break when x_nb is reached
            if i == dtmg.x_nb:
                break
            # Update y_mean and y_std
            y_i_np = self._record_y(y_i, y_idx, dtmg.x_bsz)
            y_delta = y_i_np - y_mean
            y_mean += np.sum(y_delta, axis=0) / n_x_count
            y_delta_2 = y_i_np - y_mean
            y_std += np.sum(y_delta * y_delta_2, axis=0)
            # Prepare x
            # Send x to the device
            x_i = x_i.to(self.device)
            # Embed discrete inputs
            x_i = self._emb((x_i,))[0]
            # Repeat x along batch dimension
            x_i = x_i.repeat(
                *((x_0_batch_size * n_iter,) + (1,) * (x_i.dim() - 1))
            )
            # Compute integrated gradients
            y_0_i, y_r_i, bs_i = self._bsl_shap_per_x(
                dtmg, x_i, n_iter, x_0_seed
            )
            # Update y_r_mean and y_r_std
            y_r_delta = y_r_i - y_r_mean
            y_r_mean += np.sum(y_r_delta, axis=0) / n_x_count
            y_r_std += np.sum(y_r_delta * (y_r_i - y_r_mean), axis=0)
            # Update correlation
            corr += np.sum(y_r_delta * y_delta_2, axis=0)
            # Update BSC
            bsc_delta = bs_i - bsc_mean
            bsc_mean += np.sum(bsc_delta, axis=0) / n_x_count
            bsc += np.sum(
                bsc_delta
                * y_delta_2[(...,) + (None,) * len(self.embedding_size[0])],
                axis=0,
            )
            # Check IG error and display incremental value in tqdm
            if check_error:
                bs_sum_i = np.sum(
                    bs_i.reshape((dtmg.x_bsz, dtmg.n_y_idx, -1)), axis=2
                )
                bs_error_i = np.mean(np.abs(bs_sum_i - y_r_i + y_0_i), axis=1)
                bs_error += np.sum(bs_error_i - bs_error) / n_x_count
                tqdm_iterator.set_postfix_str(
                    f"bs err: {bs_error:>9.6f}", refresh=False
                )
        # Finalize y_std and y_r_std
        y_std /= dtmg.n_x
        y_r_std /= dtmg.n_x
        y_y_r_std = np.sqrt(y_std * y_r_std)
        # Finalize BSC
        bsc /= dtmg.n_x
        bsc /= y_y_r_std[(...,) + (None,) * (bsc.ndim - 1)]
        # Check BSC error
        if check_error:
            bsc_sum = np.sum(np.reshape(bsc, (dtmg.n_y_idx, -1)), axis=1)
            corr /= dtmg.n_x
            corr /= y_y_r_std
            print(f"bsc err: {np.mean(np.abs(bsc_sum - corr)):>9.6f}")
        # Return results
        return bsc

    @torch.no_grad()
    def error(self, bsc, y_idx=None, batch_size=None, x_seed=None, n_x=None):
        """
        Compute BSC error.

        Parameters
        ----------
        bsc : ArrayLike
            BSC attributions of shape (:attr:`n_y_idx`, * unbatched :obj:`x`
            shape)
        y_idx : None | int | ArrayLike
            Selected output component indices. If :obj:`None`, :obj:`y_idx` is
            resolved to all output component indices.
        batch_size : int
            Set :attr:`x_bsz`.
        x_seed : None | int
            Seed associated with :attr:`x_dtld`.
        n_x : None | int
            - None : :attr:`x_dtld` iterates over the whole dataset.
            - int : Number of :obj:`x` inputs sampled from the dataset.

        Returns
        -------
        ArrayLike
            Per output component mean absolute error of BSC approximations.
            For each output component, the *completeness* property of BSC states
            that the sum of input component attributions must be equal to the
            correlation between model predictions and *true* outputs.
        """
        # Check for multi_x
        assert (
            not self.multi_x
        ), "Baseline Shapley does not support multiple inputs."
        # Set module to eval mode
        self.module.eval()
        # Init data manager
        dtmg = DataManager(self)
        y_idx = dtmg.add_data_iter_x(n_x, y_idx, batch_size, x_seed)
        # Init outputs
        y_mean = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_std = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_r_mean = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_r_std = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        corr = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        # Iterate over x
        for i, (x_i, y_i) in enumerate(
            tqdm(dtmg.x_dtld, total=dtmg.x_nb, desc="igc err")
        ):
            n_x_count = (i + 1) * dtmg.x_bsz
            # Break when x_nb is reached
            if i == dtmg.x_nb:
                break
            # Update y_mean and y_std
            y_i_np = self._record_y(y_i, y_idx, dtmg.x_bsz)
            y_delta = y_i_np - y_mean
            y_mean += np.sum(y_delta, axis=0) / n_x_count
            y_delta_2 = y_i_np - y_mean
            y_std += np.sum(y_delta * y_delta_2, axis=0)
            # Send x to the device
            x_i = x_i.to(self.device)
            # Embed discrete inputs
            x_i = self._emb((x_i,))
            # Compute predictions
            y_r_i = self._fwd_no_grad(x_i)
            # Update y_r_mean and y_r_std
            y_r_i_np = self._record_y(y_r_i, y_idx, dtmg.x_bsz)
            y_r_delta = y_r_i_np - y_r_mean
            y_r_mean += np.sum(y_r_delta, axis=0) / n_x_count
            y_r_std += np.sum(y_r_delta * (y_r_i_np - y_r_mean), axis=0)
            # Update correlation
            corr += np.sum(y_r_delta * y_delta_2, axis=0)
        # Finalize y_std and y_r_std
        y_std /= dtmg.n_x
        y_r_std /= dtmg.n_x
        y_y_r_std = np.sqrt(y_std * y_r_std)
        # Finalize correlation
        corr /= dtmg.n_x
        corr /= y_y_r_std
        # Check BSC error
        bsc_sum = np.sum(np.reshape(bsc, (dtmg.n_y_idx, -1)), axis=1)
        error = np.abs(bsc_sum - corr)
        print(f"bsc err: {np.mean(error):>9.6f}")
        return error
