"""
Integrated Gradient Correlation (IGC) utilities.

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


class Gradients(AbstractAttributionMethod):
    """
    Gradients.

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

    def compute(  # pylint: disable=W0221
        self, x, y_idx=None, batch_size=None, x_seed=None
    ):
        """
        Compute gradients.

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
        tuple
            - ArrayLike | tuple(ArrayLike) : sampled inputs
            - ArrayLike : corresponding *true* outputs
            - ArrayLike : model predictions
            - ArrayLike | tuple(ArrayLike) : gradients of shape (:attr:`n_x`, :attr:`n_y_idx`, * unbatched :obj:`x` shape)
        """
        # Set module to eval mode
        self.module.eval()
        # Init data manager
        dtmg = DataManager(self, y_required=False)
        y_idx = dtmg.add_data_iter_x_y_idx(x, y_idx, batch_size, x_seed)
        # Init outputs
        x_np = self._init_output(
            (dtmg.n_x,), self.x_size, dtmg.get_x_dtype(numpy=True)
        )
        y_np = np.zeros((dtmg.n_x, dtmg.n_y_idx), dtype=self.dtype_np)
        y_r = np.zeros((dtmg.n_x, dtmg.n_y_idx), dtype=self.dtype_np)
        grad = self._init_output((dtmg.n_x, dtmg.n_y_idx), self.embedding_size)
        # Iterate over x #######################################################
        for i, (x_i, y_i) in enumerate(
            tqdm(dtmg.x_dtld, total=dtmg.x_nb, desc="grad")
        ):
            # Break when x_nb is reached
            if i == dtmg.x_nb:
                break
            # Multi x
            if not self.multi_x:
                x_i = (x_i,)
            # Current x slice
            x_slc = slice(i * dtmg.x_bsz, (i + 1) * dtmg.x_bsz)
            # Record x, y
            for k, x_i_k in enumerate(x_i):
                x_np[k][x_slc] += x_i_k.cpu().numpy()
            y_i_np = self._record_y(y_i, y_idx, dtmg.x_bsz)
            if y_i_np is not None:
                y_np[x_slc] += y_i_np
            # Prepare x
            with torch.no_grad():
                # Send x to the device
                x_i = tuple(x_i_k.to(self.device) for x_i_k in x_i)
                # Embed discrete inputs
                x_i = self._emb(x_i)
                # Repeat x along batch dimension
                x_i = tuple(
                    x_i_k.repeat(
                        *((dtmg.y_idx_bsz,) + (1,) * (x_i_k.dim() - 1))
                    )
                    for x_i_k in x_i
                )
            # Forward pass
            y_f = self._fwd(x_i)
            # Iterate over output features y_idx ###############################
            for j, y_idx_j in enumerate(dtmg.y_idx_dtld):
                # Current slice and batchsize
                y_slc = slice(
                    j * dtmg.y_idx_bsz,
                    min(dtmg.n_y_idx, (j + 1) * dtmg.y_idx_bsz),
                )
                batch_size = y_slc.stop - y_slc.start
                # Compute predictions and gradients
                y_r_i_j, grad_i_j = self._bwd(x_i, y_f, y_idx_j)
                y_r_i_j = y_r_i_j.reshape((dtmg.y_idx_bsz, dtmg.x_bsz))[
                    :batch_size
                ]
                grad_i_j = tuple(
                    grad_i_j_k.reshape(
                        (dtmg.y_idx_bsz, dtmg.x_bsz) + grad_i_j_k.shape[1:]
                    )[:batch_size]
                    for grad_i_j_k in grad_i_j
                )
                # Record results
                y_r[x_slc, y_slc] += y_r_i_j.T
                for k, grad_i_j_k in enumerate(grad_i_j):
                    grad[k][x_slc, y_slc] += grad_i_j_k.swapaxes(0, 1)
        # Return results
        if self.multi_x:
            return x_np, y_np, y_r, grad
        return x_np[0], y_np, y_r, grad[0]


class IntegratedGradients(AbstractAttributionMethod):
    """
    Integrated Gradients (IG).

    See the original paper :cite:`SundararajanAxiomaticAttributionDeep2017` for
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

    def __init__(
        self,
        module,
        dataset=None,
        dtld_kwargs=None,
        forward_method_name=None,
        forward_method_kwargs=None,
        dtype=torch.float32,
        dtype_cat=torch.int32,
    ):
        super().__init__(
            module,
            dataset,
            dtld_kwargs,
            forward_method_name,
            forward_method_kwargs,
            dtype,
            dtype_cat,
        )
        # Init other parameters
        self.w = None
        self.use_z = False
        self.z_size = None
        self.final_lin_weight = None
        self.final_lin_bias = None
        self.ig_post_func = None
        self.ig_post_func_kwargs = {}
        self.ig_post_size = self.embedding_size

    def add_embedding_method(
        self,
        embedding_method_name,
        embedding_method_kwargs=None,
        embedding_n_cat=None,
    ):
        super().add_embedding_method(
            embedding_method_name,
            embedding_method_kwargs,
            embedding_n_cat,
        )
        if self.ig_post_func is None:
            self.ig_post_size = self.embedding_size
        return self

    def _init_interpolation_coefficients(self, n_steps):
        assert n_steps > 0, "The number of steps must be positive."
        self.w = tuple(
            torch.linspace(
                0.0, 1.0, n_steps, dtype=self.dtype, device=self.device
            )[(...,) + (None,) * (1 + len(sz_i))]
            for sz_i in self.embedding_size
        )
        return self

    @torch.no_grad()
    def _get_z_size_from_dtld(self):
        x, _ = self.dataset[0]
        if self.multi_x:
            x = tuple(x_i.unsqueeze(dim=0).to(self.device) for x_i in x)
        else:
            x = (x.unsqueeze(dim=0).to(self.device),)
        return self._fwd(self._emb(x)).size(1)

    def add_final_linear_layer(self, final_linear_layer):
        """
        Add a final linear layer, separated from the forward method.

        It accelerates the computation of Integrated Gradients (IG) when the
        output dimensionality of :obj:`y` is large compared to the size of the
        latent variable :obj:`z` employed before this final linear layer.

        .. warning::
            The effect of this layer must be excluded from the forward method
            defined by :attr:`forward_method_name` at initialization.

        Parameters
        ----------
        final_linear_layer : torch.nn.Linear | str
            Final linear layer of the :attr:`module`. It can also be defined by
            its name.

        Returns
        -------
        self
        """
        self.use_z = True
        # Find and check the final linear layer
        if isinstance(final_linear_layer, str):
            for name, layer in self.module.named_modules():
                if name == final_linear_layer:
                    final_linear_layer = layer
                    break
        assert isinstance(
            final_linear_layer, nn.Linear
        ), "Final linear layer must be inherited from torch.nn.Linear."
        # Extract weight and bias parameters
        self.final_lin_weight = final_linear_layer.weight.detach().cpu().numpy()
        if final_linear_layer.bias is not None:
            self.final_lin_bias = final_linear_layer.bias.detach().cpu().numpy()
            self.final_lin_bias = self.final_lin_bias[:, None]
        # Check z_size
        z_size = self._get_z_size_from_dtld()
        self.z_size = self._check_y_size(z_size)
        return self

    def _apply_final_lin(self, z, use_bias=True):
        out_dim, in_dim = self.final_lin_weight.shape
        out_shape = (out_dim,) + z.shape[1:]
        y = np.matmul(self.final_lin_weight, z.reshape((in_dim, -1)))
        if use_bias and self.final_lin_bias is not None:
            y += self.final_lin_bias
        return y.reshape(out_shape)

    def _apply_final_lin_error(self, z):
        y = np.dot(z.cpu().numpy(), self.final_lin_weight.T)
        if self.final_lin_bias is not None:
            y += self.final_lin_bias[:, 0]
        return torch.as_tensor(y)

    @torch.no_grad()
    def _get_ig_post_size_from_dtld(self):
        x, _ = self.dataset[0]
        if self.multi_x:
            x = tuple(x_i.unsqueeze(dim=0).to(self.device) for x_i in x)
        else:
            x = (x.unsqueeze(dim=0).to(self.device),)
        x_emb = self._emb(x)
        ig_ = tuple(x_emb_i.unsqueeze(dim=1).cpu().numpy() for x_emb_i in x_emb)
        x_ = tuple(x_i.cpu().numpy() for x_i in x)
        ig_p = self._ig_post(ig_, x_)
        if isinstance(ig_p, (tuple, list)):
            return tuple(ig_p_i.shape[2:] for ig_p_i in ig_p)
        return (ig_p.shape[2:],)

    def add_ig_post_function(self, ig_post_func, ig_post_func_kwargs=None):
        """
        Add a function to postprocess individual IG attributions.

        .. note::
            Adding a function to postprocess individual IG modifies
            the output shapes of computed attributions.

        .. warning::
            The IG postprocessing function must have the following signature:

            .. code-block:: python

                def ig_post(ig_x_1, ..., ig_x_n, x_1, ..., x_n, **kwargs)
                    # Do something on IG data
                    return ig_x_1, ..., ig_x_n

        Parameters
        ----------
        ig_post_func : function
            Function to postprocess individual IG attributions.
        ig_post_func_kwargs : dict
            Additional keyword arguments to the IG postprocessing function.

        Returns
        -------
        self
        """
        self.ig_post_func = ig_post_func
        self.ig_post_func_kwargs = self._check_kwargs(ig_post_func_kwargs)
        # Check ig_post_size
        ig_post_size = self._get_ig_post_size_from_dtld()
        self.ig_post_size = self._check_x_size(ig_post_size)
        return self

    def _ig_post(self, ig_x, x):
        if self.ig_post_func is None:
            return ig_x
        # pylint: disable=E1102
        return self.ig_post_func(*ig_x, *x, **self.ig_post_func_kwargs)

    def _int_grad_per_x_per_x_0(self, dtmg, x, x_0, n_steps):
        # Init outputs
        y_0 = np.zeros((dtmg.n_y_idx, dtmg.x_bsz), dtype=self.dtype_np)
        y_r = np.zeros((dtmg.n_y_idx, dtmg.x_bsz), dtype=self.dtype_np)
        int_grad = self._init_output(
            (dtmg.n_y_idx, dtmg.x_bsz), self.embedding_size
        )
        # Generate inputs along a linear path between x_0 and x
        with torch.no_grad():
            x_s = tuple()
            for x_0_i, x_i, w_i in zip(x_0, x, self.w):
                x_s_i = (1.0 - w_i) * x_0_i.unsqueeze(
                    dim=0
                ) + w_i * x_i.unsqueeze(dim=0)
                x_s += (x_s_i.flatten(0, 1),)
        # Compute input/baseline differences
        x_diff = tuple(
            (x_i - x_0_i)
            .cpu()
            .numpy()
            .reshape((dtmg.y_idx_bsz, dtmg.x_0_bsz, dtmg.x_bsz) + sz_i)
            for x_i, x_0_i, sz_i in zip(x, x_0, self.embedding_size)
        )
        # Forward pass
        y_f = self._fwd(x_s)
        # Iterate over output features y_idx
        for i, y_idx_i in enumerate(dtmg.y_idx_dtld):
            # Current slice and batchsize
            y_slc = slice(
                i * dtmg.y_idx_bsz, min(dtmg.n_y_idx, (i + 1) * dtmg.y_idx_bsz)
            )
            batch_size = y_slc.stop - y_slc.start
            # Compute predictions and gradients
            y_r_i, grad_i = self._bwd(x_s, y_f, y_idx_i)
            y_r_i = y_r_i.reshape(
                (n_steps, dtmg.y_idx_bsz, dtmg.x_0_bsz, dtmg.x_bsz)
            )[:, :batch_size]
            grad_i = tuple(
                grad_i_j.reshape(
                    (n_steps, dtmg.y_idx_bsz, dtmg.x_0_bsz, dtmg.x_bsz) + sz_j
                )[:, :batch_size]
                for grad_i_j, sz_j in zip(grad_i, self.embedding_size)
            )
            # Record y_0 and y_r, and sum over baselines
            y_0[y_slc] += y_r_i[0].sum(axis=1)
            y_r[y_slc] += y_r_i[-1].sum(axis=1)
            # Compute integrated gradients (Riemann sums, trapezoidal rule)
            for j, (grad_i_j, x_diff_j) in enumerate(zip(grad_i, x_diff)):
                int_grad_i_j = grad_i_j[:-1] + grad_i_j[1:]
                int_grad_i_j = 0.5 * np.mean(int_grad_i_j, axis=0)
                int_grad_i_j *= x_diff_j[:batch_size]
                int_grad[j][y_slc] += int_grad_i_j.sum(axis=1)
        return y_0, y_r, int_grad

    def _int_grad_per_x_per_x_0_optim(self, dtmg, x, x_0, n_steps):
        # Init outputs
        z_0 = np.zeros((dtmg.n_z_idx, dtmg.x_bsz), dtype=self.dtype_np)
        z_r = np.zeros((dtmg.n_z_idx, dtmg.x_bsz), dtype=self.dtype_np)
        int_grad = self._init_output(
            (dtmg.n_z_idx, dtmg.x_bsz), self.embedding_size
        )
        # Generate inputs along a linear path between x_0 and x
        with torch.no_grad():
            x_s = tuple()
            for x_0_i, x_i, w_i in zip(x_0, x, self.w):
                x_s_i = (1.0 - w_i) * x_0_i.unsqueeze(
                    dim=0
                ) + w_i * x_i.unsqueeze(dim=0)
                x_s += (x_s_i.flatten(0, 1),)
        # Compute input/baseline differences
        x_diff = tuple(
            (x_i - x_0_i)
            .cpu()
            .numpy()
            .reshape((dtmg.z_idx_bsz, dtmg.x_0_bsz, dtmg.x_bsz) + sz_i)
            for x_i, x_0_i, sz_i in zip(x, x_0, self.embedding_size)
        )
        # Forward pass
        z_f = self._fwd(x_s)
        # Iterate over output features y_idx
        for i, z_idx_i in enumerate(dtmg.z_idx_dtld):
            # Current slice and batchsize
            z_slc = slice(
                i * dtmg.z_idx_bsz, min(dtmg.n_z_idx, (i + 1) * dtmg.z_idx_bsz)
            )
            batch_size = z_slc.stop - z_slc.start
            # Compute predictions and gradients
            z_r_i, grad_i = self._bwd(x_s, z_f, z_idx_i)
            z_r_i = z_r_i.reshape(
                (n_steps, dtmg.z_idx_bsz, dtmg.x_0_bsz, dtmg.x_bsz)
            )[:, :batch_size]
            grad_i = tuple(
                grad_i_j.reshape(
                    (n_steps, dtmg.z_idx_bsz, dtmg.x_0_bsz, dtmg.x_bsz) + sz_j
                )[:, :batch_size]
                for grad_i_j, sz_j in zip(grad_i, self.embedding_size)
            )
            # Record z_0 and z_r, and sum over baselines
            z_0[z_slc] += z_r_i[0].sum(axis=1)
            z_r[z_slc] += z_r_i[-1].sum(axis=1)
            # Compute integrated gradients (Riemann sums, trapezoidal rule)
            for j, (grad_i_j, x_diff_j) in enumerate(zip(grad_i, x_diff)):
                int_grad_i_j = grad_i_j[:-1] + grad_i_j[1:]
                int_grad_i_j = 0.5 * np.mean(int_grad_i_j, axis=0)
                int_grad_i_j *= x_diff_j[:batch_size]
                int_grad[j][z_slc] += int_grad_i_j.sum(axis=1)
        # Apply final linear layer
        y_0 = self._apply_final_lin(z_0)
        y_r = self._apply_final_lin(z_r)
        int_grad = tuple(
            self._apply_final_lin(int_grad_i, use_bias=False)
            for int_grad_i in int_grad
        )
        return y_0, y_r, int_grad

    def _int_grad_per_x(self, dtmg, x, n_steps):
        # Init outputs
        y_0 = np.zeros((dtmg.x_bsz, dtmg.n_y_idx), dtype=self.dtype_np)
        y_r = np.zeros((dtmg.x_bsz, dtmg.n_y_idx), dtype=self.dtype_np)
        int_grad = self._init_output(
            (dtmg.x_bsz, dtmg.n_y_idx), self.embedding_size
        )
        # Define x_0 repeat size
        if self.use_z:
            x_0_rep = dtmg.z_idx_bsz
        else:
            x_0_rep = dtmg.y_idx_bsz
        # Iterate over baselines
        for i, (x_0_i, _) in enumerate(dtmg.x_0_dtld):
            # Break when x_0_nb is reached
            if i == dtmg.x_0_nb:
                break
            # Multi x
            if not self.multi_x:
                x_0_i = (x_0_i,)
            # Prepare x_0
            with torch.no_grad():
                # Send x_0 to the device
                x_0_i = tuple(x_0_i_j.to(self.device) for x_0_i_j in x_0_i)
                # Embed discrete inputs
                x_0_i = self._emb(x_0_i)
                # Repeat x_0 along batch dimension
                x_0_i = tuple(
                    x_0_i_j.repeat(*((x_0_rep,) + (1,) * (x_0_i_j.dim() - 1)))
                    for x_0_i_j in x_0_i
                )
            # Compute integrated gradients (summed over baselines)
            if self.use_z:
                y_0_i, y_r_i, int_grad_i = self._int_grad_per_x_per_x_0_optim(
                    dtmg, x, x_0_i, n_steps
                )
            else:
                y_0_i, y_r_i, int_grad_i = self._int_grad_per_x_per_x_0(
                    dtmg, x, x_0_i, n_steps
                )
            # Record y_0 and y_r
            y_0 += y_0_i.T
            y_r += y_r_i.T
            # Record integrated gradients
            for j, int_grad_i_j in enumerate(int_grad_i):
                int_grad[j][...] += int_grad_i_j.swapaxes(0, 1)
        # Average across baselines
        y_0 /= dtmg.n_x_0
        y_r /= dtmg.n_x_0
        for int_grad_i in int_grad:
            int_grad_i /= dtmg.n_x_0
        return y_0, y_r, int_grad

    def compute(  # pylint: disable=W0221
        self,
        x,
        x_0=None,
        y_idx=None,
        n_steps=64,
        batch_size=None,
        x_seed=None,
        x_0_seed=100,
        check_error=True,
    ):
        """
        Compute Integrated Gradients.

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
        check_error : bool
            If :obj:`True`, the mean absolute error of IG approximations is
            reported. For each input, baseline, and output component, the
            *completeness* property of IG states that the sum of input component
            attributions must be equal to the difference between the model
            predictions associated with the input and baseline under scrutiny.

        Returns
        -------
        tuple
            - ArrayLike | tuple(ArrayLike) : sampled inputs
            - ArrayLike : corresponding *true* outputs
            - ArrayLike : model predictions of sampled inputs
            - ArrayLike : model predictions of baselines
            - ArrayLike | tuple(ArrayLike) : IG attributions of shape (:attr:`n_x`, :attr:`n_y_idx`, * unbatched :obj:`x` shape).
        """
        # Set module to eval mode
        self.module.eval()
        # Init data manager
        dtmg = DataManager(self, y_required=False)
        y_idx = dtmg.add_data(
            x, x_0, y_idx, n_steps, batch_size, x_seed, x_0_seed
        )
        # Init interpolation coefficients w
        self._init_interpolation_coefficients(n_steps)
        # Init outputs
        x_np = self._init_output(
            (dtmg.n_x,), self.x_size, dtmg.get_x_dtype(numpy=True)
        )
        y_np = np.zeros((dtmg.n_x, dtmg.n_y_idx), dtype=self.dtype_np)
        y_0 = np.zeros((dtmg.n_x, dtmg.n_y_idx), dtype=self.dtype_np)
        y_r = np.zeros((dtmg.n_x, dtmg.n_y_idx), dtype=self.dtype_np)
        int_grad = self._init_output(
            (dtmg.n_x, dtmg.n_y_idx), self.ig_post_size
        )
        # Define x repeat size
        if self.use_z:
            x_rep = dtmg.x_0_bsz * dtmg.z_idx_bsz
        else:
            x_rep = dtmg.x_0_bsz * dtmg.y_idx_bsz
        # Iterate over x
        for i, (x_i, y_i) in enumerate(
            tqdm(dtmg.x_dtld, total=dtmg.x_nb, desc="ig")
        ):
            # Break when x_nb is reached
            if i == dtmg.x_nb:
                break
            # Multi x
            if not self.multi_x:
                x_i = (x_i,)
            # Current slice
            slc = slice(i * dtmg.x_bsz, (i + 1) * dtmg.x_bsz)
            # Record x, y
            for j, x_i_j in enumerate(x_i):
                x_np[j][slc] += x_i_j.cpu().numpy()
            y_i_np = self._record_y(y_i, y_idx, dtmg.x_bsz)
            if y_i_np is not None:
                y_np[slc] += y_i_np
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
            # Record y_0 and y_r
            y_0[slc] += y_0_i
            y_r[slc] += y_r_i
            # Apply integrated gradients post-function
            ig_i = self._ig_post(ig_i, tuple(x_np_i[slc] for x_np_i in x_np))
            # Record integrated gradients
            for j, ig_i_j in enumerate(ig_i):
                int_grad[j][slc] += ig_i_j
        # Check error
        if check_error:
            int_grad_sum = np.zeros(
                (dtmg.n_x, dtmg.n_y_idx), dtype=self.dtype_np
            )
            for ig_i in int_grad:
                int_grad_sum += np.sum(
                    ig_i.reshape((dtmg.n_x, dtmg.n_y_idx, -1)), axis=2
                )
            print(f"ig err: {np.mean(np.abs(int_grad_sum - y_r + y_0)):>9.6f}")
        # Return results
        if self.multi_x:
            return x_np, y_np, y_0, y_r, int_grad
        return x_np[0], y_np, y_0, y_r, int_grad[0]


class IntGradCorr(IntegratedGradients):
    """
    Integrated Gradient Correlation (IGC).

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
        Compute Integrated Gradient Correlation (IGC).

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
            If :obj:`True`, the mean absolute error of IG and IGC approximations
            is reported. For each input, baseline, and output component, the
            *Completeness* property of IG states that the sum of input component
            attributions must be equal to the difference between the model
            predictions associated with the input and baseline under scrutiny.
            For each output component, the *completeness* property of IGC states
            that the sum of input component attributions must be equal to the
            correlation between model predictions and *true* outputs.

        Returns
        -------
        ArrayLike | tuple(ArrayLike)
            IGC attributions of shape (:attr:`n_y_idx`, * unbatched :obj:`x`
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
        y_mean = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_std = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_r_mean = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        y_r_std = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        corr = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        igc = self._init_output((dtmg.n_y_idx,), self.ig_post_size)
        igc_mean = self._init_output((dtmg.n_y_idx,), self.ig_post_size)
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
            dtmg.x_dtld, total=dtmg.x_nb, desc="igc", postfix=postfix
        )
        for i, (x_i, y_i) in enumerate(tqdm_iterator):
            n_x_count = (i + 1) * dtmg.x_bsz
            # Break when x_nb is reached
            if i == dtmg.x_nb:
                break
            # Multi x
            if not self.multi_x:
                x_i = (x_i,)
            # Update y_mean and y_std
            y_i_np = self._record_y(y_i, y_idx, dtmg.x_bsz)
            y_delta = y_i_np - y_mean
            y_mean += np.sum(y_delta, axis=0) / n_x_count
            y_delta_2 = y_i_np - y_mean
            y_std += np.sum(y_delta * y_delta_2, axis=0)
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
            y_r_std += np.sum(y_r_delta * (y_r_i - y_r_mean), axis=0)
            # Update correlation
            corr += np.sum(y_r_delta * y_delta_2, axis=0)
            # Apply IG post-function
            ig_i = self._ig_post(ig_i, x_i_np)
            # Update IGC
            for j, (ig_i_j, sz_j) in enumerate(zip(ig_i, self.ig_post_size)):
                igc_delta = ig_i_j - igc_mean[j]
                igc_mean[j][...] += np.sum(igc_delta, axis=0) / n_x_count
                igc[j][...] += np.sum(
                    igc_delta * y_delta_2[(...,) + (None,) * len(sz_j)], axis=0
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
        # Finalize y_std and y_r_std
        y_std /= dtmg.n_x
        y_r_std /= dtmg.n_x
        y_y_r_std = np.sqrt(y_std * y_r_std)
        # Finalize IGC
        for igc_i in igc:
            igc_i /= dtmg.n_x
            igc_i /= y_y_r_std[(...,) + (None,) * (igc_i.ndim - 1)]
        # Check IGC error
        if check_error:
            igc_sum = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
            for igc_i in igc:
                igc_sum += np.sum(np.reshape(igc_i, (dtmg.n_y_idx, -1)), axis=1)
            corr /= dtmg.n_x
            corr /= y_y_r_std
            print(f"igc err: {np.mean(np.abs(igc_sum - corr)):>9.6f}")
        # Return results
        if self.multi_x:
            return igc
        return igc[0]

    @torch.no_grad()
    def error(self, igc, y_idx=None, batch_size=None, x_seed=None, n_x=None):
        """
        Compute IGC error.

        Parameters
        ----------
        igc : ArrayLike | tuple(ArrayLike)
            IGC attributions of shape (:attr:`n_y_idx`, * unbatched :obj:`x`
            shape)
        y_idx : None | int | ArrayLike
            Selected output component indices. If :obj:`None`, :obj:`y_idx` is
            resolved to all output component indices.
        batch_size : None | int
            - None : Set :attr:`x_bsz` = 1.
            - int : Set :attr:`x_bsz`.
        x_seed : None | int
            Seed associated with :attr:`x_dtld`.
        n_x : None | int
            - None : :attr:`x_dtld` iterates over the whole dataset.
            - int : Number of :obj:`x` inputs sampled from the dataset.

        Returns
        -------
        ArrayLike
            Per output component mean absolute error of IGC approximations.
            For each output component, the *completeness* property of IGC states
            that the sum of input component attributions must be equal to the
            correlation between model predictions and *true* outputs.
        """
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
            # Multi x
            if not self.multi_x:
                x_i = (x_i,)
            # Update y_mean and y_std
            y_i_np = self._record_y(y_i, y_idx, dtmg.x_bsz)
            y_delta = y_i_np - y_mean
            y_mean += np.sum(y_delta, axis=0) / n_x_count
            y_delta_2 = y_i_np - y_mean
            y_std += np.sum(y_delta * y_delta_2, axis=0)
            # Send inputs to the device
            x_i = tuple(x_i_j.to(self.device) for x_i_j in x_i)
            y_i = y_i.to(self.device)
            # Embed discrete inputs
            x_i = self._emb(x_i)
            # Compute predictions
            y_r_i = self._fwd_no_grad(x_i)
            if self.use_z:
                y_r_i = self._apply_final_lin_error(y_r_i)
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
        # Multi x
        if not self.multi_x:
            igc = (igc,)
        # Check IGC error
        igc_sum = np.zeros(dtmg.n_y_idx, dtype=self.dtype_np)
        for igc_i in igc:
            igc_sum += np.sum(np.reshape(igc_i, (dtmg.n_y_idx, -1)), axis=1)
        error = np.abs(igc_sum - corr)
        print(f"igc err: {np.mean(error):>9.6f}")
        return error
