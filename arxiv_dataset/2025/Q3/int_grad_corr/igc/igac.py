"""
Integrated Gradient Auto-Correlation (IGaC) utilities.

.. note::
    Welford's online algorithm :cite:`WelfordNoteMethodCalculating1962`
    is used for the computation of mean, standard deviation, and correlation
    statistics (see `wikipedia`_).

.. _wikipedia: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
"""

import numpy as np
import torch
from tqdm import tqdm

from .base import DataManager
from .igc import IntegratedGradients


class IntGradAutoCorr(IntegratedGradients):
    """
    Integrated Gradient Auto-Correlation (IGaC).

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

    Notes
    -----

    .. warning::
        When computing attributions on models using multiple inputs, e.g., x_1,
        x_2, and x_cat, with x_cat a categorical input, the dataset must return
        all inputs packed in a tuple, such as: (x_1, x_2, x_cat), y. Note that
        categorical inputs must be placed at the end of the tuple.
    """

    def compute(  # pylint: disable=W0221,W0237
        self,
        x_0=None,
        y_idx=None,
        n_steps=64,
        batch_size=None,
        x_seed=None,
        x_0_seed=100,
        n_x=None,
        check_error=True,
    ):
        """
        Compute Integrated Gradient Auto-Correlation (IGaC).

        Parameters
        ----------
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
        n_x : None | int
            - None : :attr:`x_dtld` iterates over the whole dataset.
            - int : Number of :obj:`x` inputs sampled from the dataset.
        check_error : bool
            If :obj:`True`, the mean absolute error of IG and IGaC
            approximations is reported. For each input, baseline, and output
            component, the *Completeness* property of IG states that the sum of
            input component attributions must be equal to the difference between
            the model predictions associated with the input and baseline under
            scrutiny. For each output component, the *completeness* property of
            IGaC states that the sum of input component attributions must be
            equal to 1.

        Returns
        -------
        ArrayLike | tuple(ArrayLike)
            IGaC attributions of shape (:attr:`n_y_idx`, * unbatched :obj:`x`
            shape)
        """
        # Set module to eval mode
        self.module.eval()
        # Init data manager
        dtmg = DataManager(self)
        y_idx = dtmg.add_data(
            n_x, x_0, y_idx, n_steps, batch_size, x_seed, x_0_seed
        )
        # Init interpolation coefficients w
        self._init_interpolation_coefficients(n_steps)
        # Init outputs
        ig_error = 0.0
        y_r_mean = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_r_var = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        igac = self._init_output((dtmg.n_y_idx,), self.ig_post_size)
        igac_mean = self._init_output((dtmg.n_y_idx,), self.ig_post_size)
        # Define x repeat size
        if self.use_z:
            x_rep = dtmg.x_0_bsz * dtmg.z_idx_bsz
        else:
            x_rep = dtmg.x_0_bsz * dtmg.y_idx_bsz
        # Iterate over x
        postfix = None
        if check_error:
            postfix = "ig err: ?"
        tqdm_iterator = tqdm(
            dtmg.x_dtld, total=dtmg.x_nb, desc="igac", postfix=postfix
        )
        for i, (x_i, _) in enumerate(tqdm_iterator):
            n_x_count = (i + 1) * dtmg.x_bsz
            # Break when x_nb is reached
            if i == dtmg.x_nb:
                break
            # Multi x
            if not self.multi_x:
                x_i = (x_i,)
            # Record original x
            x_i_np = (x_i_j.cpu().numpy() for x_i_j in x_i)
            # Prepare x
            with torch.no_grad():
                # Send x to the device
                x_i = tuple(x_i_j.to(self.device) for x_i_j in x_i)
                # Embed discrete inputs
                x_i = self._emb(x_i)
                # Repeat x along batch dimension
                x_i = tuple(
                    x_i_j.repeat(*((x_rep,) + (1,) * (x_i_j.dim() - 1)))
                    for x_i_j in x_i
                )
            # Update x_0_dtld seed
            dtmg.update_x_0_dtld_seed()
            # Compute integrated gradients
            y_0_i, y_r_i, ig_i = self._int_grad_per_x(dtmg, x_i, n_steps)
            # Update y_r_mean and y_r_std
            y_r_delta = y_r_i - y_r_mean
            y_r_mean += np.sum(y_r_delta, axis=0) / n_x_count
            y_r_delta_2 = y_r_i - y_r_mean
            y_r_var += np.sum(y_r_delta * y_r_delta_2, axis=0)
            # Apply IG post-function
            ig_i = self._ig_post(ig_i, x_i_np)
            # Update IGaC
            for j, (ig_i_j, sz_j) in enumerate(zip(ig_i, self.ig_post_size)):
                igac_delta = ig_i_j - igac_mean[j]
                igac_mean[j][...] += np.sum(igac_delta, axis=0) / n_x_count
                igac[j][...] += np.sum(
                    igac_delta * y_r_delta_2[(...,) + (None,) * len(sz_j)],
                    axis=0,
                )
            # Check IG error and display incremental value in tqdm
            if check_error:
                ig_sum_i = np.zeros(
                    (dtmg.x_bsz, dtmg.n_y_idx), dtype=self.dtype_np
                )
                for ig_i_j in ig_i:
                    ig_sum_i += np.sum(
                        ig_i_j.reshape((dtmg.x_bsz, dtmg.n_y_idx, -1)), axis=2
                    )
                ig_error_i = np.mean(np.abs(ig_sum_i - y_r_i + y_0_i), axis=1)
                ig_error += np.sum(ig_error_i - ig_error) / n_x_count
                tqdm_iterator.set_postfix_str(
                    f"ig err: {ig_error:>9.6f}", refresh=False
                )
        # Finalize y_r_var
        y_r_var /= dtmg.n_x
        # Finalize IGaC
        for igac_i in igac:
            igac_i /= dtmg.n_x
            igac_i /= y_r_var[(...,) + (None,) * (igac_i.ndim - 1)]
        # Check IGaC error
        if check_error:
            igac_sum = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
            for igac_i in igac:
                igac_sum += np.sum(
                    np.reshape(igac_i, (dtmg.n_y_idx, -1)), axis=1
                )
            print(f"igac err: {np.mean(np.abs(igac_sum - 1.0)):>9.6f}")
        # Return results
        if self.multi_x:
            return igac
        return igac[0]

    def error(self, igac):
        """
        Compute IGaC error.

        Parameters
        ----------
        igac : ArrayLike | tuple(ArrayLike)
            IGaC attributions of shape (:attr:`n_y_idx`, * unbatched :obj:`x`
            shape)
        Returns
        -------
        ArrayLike
            Per output component mean absolute error of IGaC approximations.
            For each output component, the *completeness* property of IGaC
            states that the sum of input component attributions must be equal
            to 1.
        """
        # Multi x
        if not isinstance(igac, tuple):
            igac = (igac,)
        # Check IGaC error
        n_y = igac[0].shape[0]
        igac_sum = np.zeros(n_y, dtype=igac[0].dtype)
        for igac_i in igac:
            igac_sum += np.sum(np.reshape(igac_i, (n_y, -1)), axis=1)
        error = np.abs(igac_sum - 1.0)
        print(f"igac err: {np.mean(error):>9.6f}")
        return error
