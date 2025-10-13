"""
Naive alternatives to Integrated Gradient Correlation (IGC).

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
from scipy.stats import ttest_ind_from_stats
from tqdm import tqdm

from .base import AbstractAttributionMethod, DataManager
from .igc import IntegratedGradients


# Integrated gradients mean and std


class IntGradMeanStd(IntegratedGradients):
    """
    Mean/std of the IG distribution over the dataset (IGms).

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
        Compute mean/std of the IG distribution over the dataset (IGms).

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
            If :obj:`True`, the mean absolute error of IG approximations is
            reported. For each input, baseline, and output component, the
            *Completeness* property of IG states that the sum of input component
            attributions must be equal to the difference between the model
            predictions associated with the input and baseline under scrutiny.

        Returns
        -------
        tuple
            - ArrayLike | tuple(ArrayLike) : mean of the IG distribution over the dataset. The shape is (:attr:`n_y_idx`, * unbatched :obj:`x` shape).
            - ArrayLike | tuple(ArrayLike) : std of the IG distribution over the dataset. The shape is (:attr:`n_y_idx`, * unbatched :obj:`x` shape).
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
        ig_mean = self._init_output((dtmg.n_y_idx,), self.ig_post_size)
        ig_std = self._init_output((dtmg.n_y_idx,), self.ig_post_size)
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
            dtmg.x_dtld, total=dtmg.x_nb, desc="igms", postfix=postfix
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
            # Apply IG post-function
            ig_i = self._ig_post(ig_i, x_i_np)
            # Update IG mean and std
            for j, ig_i_j in enumerate(ig_i):
                ig_i_j_delta = ig_i_j - ig_mean[j]
                ig_mean[j][...] += np.sum(ig_i_j_delta, axis=0) / n_x_count
                ig_std[j][...] += np.sum(
                    ig_i_j_delta * (ig_i_j - ig_mean[j]), axis=0
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
        # Finalize IG std
        for ig_std_i in ig_std:
            ig_std_i /= dtmg.n_x
            ig_std_i = np.sqrt(ig_std_i)
        # Return results
        if self.multi_x:
            return ig_mean, ig_std
        return ig_mean[0], ig_std[0]


# Naive input/output correlation.


class NaiveCorrelation(AbstractAttributionMethod):
    """
    Naive component-wise input/output correlations.

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

    @torch.no_grad()
    def compute(  # pylint: disable=W0221
        self, y_idx=None, batch_size=None, x_seed=None, n_x=None
    ):
        """
        Compute the naive component-wise input/output correlations.

        Parameters
        ----------
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
        n_x : None | int
            - None : :attr:`x_dtld` iterates over the whole dataset.
            - int : Number of :obj:`x` inputs sampled from the dataset.

        Returns
        -------
        ArrayLike | tuple(ArrayLike)
            Naive component-wise input/output correlations of shape (
            :attr:`n_y_idx`, * unbatched :obj:`x` shape)
        """
        # Set module to eval mode
        self.module.eval()
        # Init data manager
        dtmg = DataManager(self)
        y_idx = dtmg.add_data_naive(n_x, y_idx, batch_size, x_seed)
        # Init outputs
        y_mean = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_std = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        x_mean = self._init_output((dtmg.n_y_idx,), self.embedding_size)
        x_std = self._init_output((dtmg.n_y_idx,), self.embedding_size)
        corr = self._init_output((dtmg.n_y_idx,), self.embedding_size)
        # Iterate over x
        for i, (x_i, y_i) in enumerate(
            tqdm(dtmg.x_dtld, total=dtmg.x_nb, desc="corr")
        ):
            n_x_count = (i + 1) * dtmg.x_bsz
            # Break when x_nb is reached
            if i == dtmg.x_nb:
                break
            # Multi x
            if not self.multi_x:
                x_i = (x_i,)
            # Record y_mean and y_std
            y_i_np = self._record_y(y_i, y_idx, dtmg.x_bsz)
            y_delta = y_i_np - y_mean
            y_mean += np.sum(y_delta, axis=0) / n_x_count
            y_delta_2 = y_i_np - y_mean
            y_std += np.sum(y_delta * y_delta_2, axis=0)
            # Send data to the device
            x_i = tuple(x_i_j.to(self.device) for x_i_j in x_i)
            y_delta_2 = torch.as_tensor(y_delta_2, device=self.device)
            # Embed discrete inputs
            x_i = self._emb(x_i)
            # Record x_mean and x_std
            x_i_delta = []
            for j, x_i_j in enumerate(x_i):
                x_i_j_np = x_i_j.unsqueeze(dim=1).repeat(
                    1, *((dtmg.n_y_idx,) + (1,) * (x_i_j.dim() - 1))
                )
                x_i_j_np = x_i_j_np.cpu().numpy()
                x_i_j_delta = x_i_j_np - x_mean[j]
                x_i_delta.append(x_i_j_delta)
                x_mean[j][...] += np.sum(x_i_j_delta, axis=0) / n_x_count
                x_std[j][...] += np.sum(
                    x_i_j_delta * (x_i_j_np - x_mean[j]), axis=0
                )
            # Prepare x_i_delta
            x_i_delta = tuple(
                torch.as_tensor(
                    x_i_j_delta, dtype=self.dtype, device=self.device
                )
                for x_i_j_delta in x_i_delta
            )
            # Iterate over output features y_idx
            for j, y_idx_j in enumerate(dtmg.y_idx_dtld):
                # Current slice and batchsize
                y_slc = slice(
                    j * dtmg.y_idx_bsz,
                    min(dtmg.n_y_idx, (j + 1) * dtmg.y_idx_bsz),
                )
                batch_size = y_slc.stop - y_slc.start
                # Compute correlation
                y_i_j_d_2 = torch.gather(
                    y_delta_2,
                    dim=1,
                    index=y_idx_j.unsqueeze(dim=0).expand(dtmg.x_bsz, -1),
                )
                for k, (x_i_k_d, sz_k) in enumerate(
                    zip(x_i_delta, self.embedding_size)
                ):
                    x_i_j_k_d = torch.gather(
                        x_i_k_d,
                        dim=1,
                        index=y_idx_j[(None, ...) + (None,) * len(sz_k)].expand(
                            dtmg.x_bsz, -1, *sz_k
                        ),
                    )
                    corr_i_j_k = (
                        x_i_j_k_d * y_i_j_d_2[(...,) + (None,) * len(sz_k)]
                    )
                    corr[k][y_slc] += (
                        torch.sum(corr_i_j_k, dim=0)[:batch_size].cpu().numpy()
                    )
        # Finalize x_std and y_std
        for x_std_i in x_std:
            x_std_i /= dtmg.n_x
            x_std_i = np.sqrt(x_std_i)
        y_std /= dtmg.n_x
        y_std = np.sqrt(y_std)
        # Compute correlation
        for corr_i, x_std_i in zip(corr, x_std):
            corr_i /= dtmg.n_x
            corr_i /= x_std_i
            corr_i /= y_std[(...,) + (None,) * (corr_i.ndim - 1)]
        # Return results
        if self.multi_x:
            return corr
        return corr[0]


# Naive two-sample t-tests


class NaiveTTest(AbstractAttributionMethod):
    """
    Naive two-sample t-tests between inputs corresponding to categories based on
    outputs characteristics.

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

    @torch.no_grad()
    def compute(  # pylint: disable=W0221
        self, cat_ranges, y_idx=None, batch_size=None, x_seed=None, n_x=None
    ):
        """
        Compute the p-values of naive two-sample t-tests between inputs
        corresponding to categories based on outputs characteristics.

        Parameters
        ----------
        cat_ranges : tuple(float)
            Tuple :obj:`(a, b)` defines a first category with entries
            :obj:`y<a` and a second category with entries :obj:`y>b`.
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
        n_x : None | int
            - None : :attr:`x_dtld` iterates over the whole dataset.
            - int : Number of :obj:`x` inputs sampled from the dataset.

        Returns
        -------
        ArrayLike | tuple(ArrayLike)
            p-values of naive two-sample t-tests of shape (:attr:`n_y_idx`, *
            unbatched :obj:`x` shape)
        """
        # Set module to eval mode
        self.module.eval()
        # Init data manager
        dtmg = DataManager(self)
        y_idx = dtmg.add_data_naive(n_x, y_idx, batch_size, x_seed)
        # Prepares cat_ranges
        assert isinstance(cat_ranges, tuple)
        if isinstance(cat_ranges[0], (float, int)):
            cat_ranges = (cat_ranges,) * dtmg.n_y_idx
        assert len(cat_ranges) == dtmg.n_y_idx
        for cat_range in cat_ranges:
            assert len(cat_range) == 2
        cat_ranges = torch.tensor(
            cat_ranges, dtype=torch.float32, device=self.device
        )
        cat_ranges = cat_ranges.transpose(0, 1).unsqueeze(dim=1)
        # Init outputs
        n_x_a = np.zeros(dtmg.n_y_idx, dtype=np.int64)
        x_mean_a = self._init_output((dtmg.n_y_idx,), self.embedding_size)
        x_std_a = self._init_output((dtmg.n_y_idx,), self.embedding_size)
        n_x_b = np.zeros(dtmg.n_y_idx, dtype=np.int64)
        x_mean_b = self._init_output((dtmg.n_y_idx,), self.embedding_size)
        x_std_b = self._init_output((dtmg.n_y_idx,), self.embedding_size)
        ttest = self._init_output((dtmg.n_y_idx,), self.embedding_size)
        # Iterate over x
        for i, (x_i, y_i) in enumerate(
            tqdm(dtmg.x_dtld, total=dtmg.x_nb, desc="ttest")
        ):
            # Break when x_nb is reached
            if i == dtmg.x_nb:
                break
            # Multi x
            if not self.multi_x:
                x_i = (x_i,)
            # Send inputs to the device
            x_i = tuple(x_i_j.to(self.device) for x_i_j in x_i)
            y_i = y_i.to(self.device)
            # Embed discrete inputs
            x_i = self._emb(x_i)
            # Iterate over output features y_idx
            for j, y_idx_j in enumerate(dtmg.y_idx_dtld):
                # Current slice and batchsize
                y_slc = slice(
                    j * dtmg.y_idx_bsz,
                    min(dtmg.n_y_idx, (j + 1) * dtmg.y_idx_bsz),
                )
                batch_size = y_slc.stop - y_slc.start
                # Extract a and b data
                y_i_j = torch.gather(
                    y_i,
                    dim=1,
                    index=y_idx_j.unsqueeze(dim=0).expand(dtmg.x_bsz, -1),
                )
                mask_a = torch.le(y_i_j, cat_ranges[0]).cpu().numpy()
                n_x_a[y_slc] += np.sum(mask_a, axis=0)[:batch_size].astype(
                    np.int64
                )
                mask_b = torch.ge(y_i_j, cat_ranges[1]).cpu().numpy()
                n_x_b[y_slc] += np.sum(mask_b, axis=0)[:batch_size].astype(
                    np.int64
                )
                for k, x_i_k in enumerate(x_i):
                    x_i_k_np = x_i_k.unsqueeze(dim=1).repeat(
                        1, *((batch_size,) + (1,) * (x_i_k.dim() - 1))
                    )
                    x_i_k_np = x_i_k_np.cpu().numpy()
                    for m in range(y_slc.start, y_slc.stop):
                        if np.sum(mask_a[:, m]):
                            selected_a = x_i_k_np[:, m][mask_a[:, m]]
                            x_delta_a = selected_a - x_mean_a[k][m]
                            x_mean_a[k][m] += (
                                np.sum(x_delta_a, axis=0) / n_x_a[m]
                            )
                            x_delta_2_a = selected_a - x_mean_a[k][m]
                            x_std_a[k][m] += np.sum(
                                x_delta_a * x_delta_2_a, axis=0
                            )
                        if np.sum(mask_b[:, m]):
                            selected_b = x_i_k_np[:, m][mask_b[:, m]]
                            x_delta_b = selected_b - x_mean_b[k][m]
                            x_mean_b[k][m] += (
                                np.sum(x_delta_b, axis=0) / n_x_b[m]
                            )
                            x_delta_2_b = selected_b - x_mean_b[k][m]
                            x_std_b[k][m] += np.sum(
                                x_delta_b * x_delta_2_b, axis=0
                            )
        # Finalize x_std_a and x_std_b (bias corrected)
        for x_std_a_i in x_std_a:
            x_std_a_i /= n_x_a[(...,) + (None,) * (x_std_a_i.ndim - 1)] - 1
            x_std_a_i = np.sqrt(x_std_a_i)
        for x_std_b_i in x_std_b:
            x_std_b_i /= n_x_b[(...,) + (None,) * (x_std_b_i.ndim - 1)] - 1
            x_std_b_i = np.sqrt(x_std_b_i)
        # Compute t-test
        for i, (m_a_i, s_a_i, m_b_i, s_b_i, sz_i) in enumerate(
            zip(x_mean_a, x_std_a, x_mean_b, x_std_b, self.embedding_size)
        ):
            for j, (m_a_ij, s_a_ij, n_a_j, m_b_ij, s_b_ij, n_b_j) in enumerate(
                zip(m_a_i, s_a_i, n_x_a, m_b_i, s_b_i, n_x_b)
            ):
                ttest_j = np.zeros(m_a_ij.size, dtype=self.dtype_np)
                for k, (m_a_ijk, s_a_ijk, m_b_ijk, s_b_ijk) in enumerate(
                    zip(
                        m_a_ij.ravel(),
                        s_a_ij.ravel(),
                        m_b_ij.ravel(),
                        s_b_ij.ravel(),
                    )
                ):
                    ttest_j[k] += ttest_ind_from_stats(
                        m_a_ijk, s_a_ijk, n_a_j, m_b_ijk, s_b_ijk, n_b_j
                    )[1]
                ttest[i][j] += ttest_j.reshape(sz_i)
        # Return results
        if self.multi_x:
            return ttest
        return ttest[0]
