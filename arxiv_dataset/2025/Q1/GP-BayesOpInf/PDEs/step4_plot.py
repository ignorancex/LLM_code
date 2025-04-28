# step4_plot.py
"""Make plots for GP-BayesOpInf PDE examples."""

__all__ = [
    "ReducedPlotter",
    "StatePlotter",
]

import dataclasses
import numpy as np
import matplotlib.pyplot as plt

import baseplots

import opinf


# Plotter classes =============================================================
@dataclasses.dataclass
class ReducedPlotter(baseplots._BasePlotter):
    """Generate plots in the reduced space.

    Parameters
    ----------
    sample_time_domains : (num_samples,) ndarray
        Time domain over which noisy snapshots are observed.
    training_time_domain : (num_regression_points,) ndarray
        Time domain for the GP estimates used in the OpInf regression.
    prediction_time_domain : (k,) ndarray
        Time domain for true states and ROM predictions.
    snapshots_compressed : (r, num_samples) ndarray
        Noisy compressed snapshots observed over ``sample_time_domain``.
    true_states_compressed : (r, k) ndarrays
        True (non-noised) states over the ``prediction_time_domain``.
    gpmeans : list of num_modes (r, num_regression_points) ndarrays
        Gaussian process means over ``training_time_domain``.
    gpstds : list of num_modes (r, num_regression_points) ndarrays
        Gaussian process standard deviations over ``training_time_domain``.
    draws_compressed : list of num_draws (r, k) ndarrays
        Collection of compressed draws over ``prediction_time_domain``.
    """

    # Properties --------------------------------------------------------------
    snapshots_compressed: np.ndarray
    true_states_compressed: np.ndarray
    gp_means: np.ndarray
    gp_stds: np.ndarray
    draws_compressed: list
    max_modes: int = 8

    @property
    def num_modes(self) -> int:
        """Number of basis functions / reduced state dimension."""
        return min(self.max_modes, self.snapshots_compressed.shape[0])

    # Main routines -----------------------------------------------------------
    def new_figure(self, figsize=None):
        """Create a new figure and subplot axes."""
        if figsize is None:
            figsize = (6, 2 * self.num_modes)
        return plt.subplots(self.num_modes, 1, figsize=figsize, sharex=True)

    def plot_gp_training_fit(self, width=3):
        """Plot the truth, the sparse / noisy / compressed data, and the GP fit
        to the data.

        Parameters
        ----------
        width : float
            Plot mean ± (``width`` * standard_deviation).
        """
        end = self.end_train_index
        fig, axes = self.new_figure()

        for i, ax in enumerate(axes.flat):
            self._plot_truth(
                ax,
                self.prediction_time_domain[:end],
                self.true_states_compressed[i, :end],
            )
            self._plot_data(
                ax,
                self.sampling_time_domain,
                self.snapshots_compressed[i],
            )
            self._plot_gp(
                ax,
                self.training_time_domain,
                self.gp_means[i],
                self.gp_stds[i],
                width=width,
            )
            ax.set_title(rf"$r = {i + 1:d}$", fontsize="large")

        fig.suptitle("GP fit", fontsize="xx-large")
        self._format_figure(fig, axes)

        return fig

    def plot_posterior(
        self,
        truth: bool = True,
        fulldomain: bool = True,
        individual: bool = False,
    ):
        """Plot the truth, sparse/noisy compressed data, and  ROM predictions.

        Parameters
        ----------
        truth : bool
            If ``True`` (default), plot the true states.
            If ``False``, do not plot the true states.
        fulldomain : bool
            If ``True`` (default), plot the true states over the full
            prediction domain.
            If ``False``, plot the true states only over the training domain.
            Ignored if ``truth=False``.
        individual : bool
            If ``True``, plot each of the draws individually.
            If ``False`` (default), plot the interquartile range of the draws.
        """
        end = None if fulldomain else self.end_train_index
        plotdraws = self._plot_draws if individual else self._plot_percentiles
        fig, axes = self.new_figure()

        for i, ax in enumerate(axes.flat):
            if truth:
                self._plot_truth(
                    ax,
                    self.prediction_time_domain[:end],
                    self.true_states_compressed[i, :end],
                )

            self._plot_data(
                ax,
                self.sampling_time_domain,
                self.snapshots_compressed[i],
            )

            draws = [draw[i] for draw in self.draws_compressed]
            plotdraws(ax, self.prediction_time_domain, draws)
            self._plot_samplemean(ax, self.prediction_time_domain, draws)

            ax.axvline(self.training_time_domain[-1], lw=1, color="black")
            ax.set_title(rf"$r = {i + 1:d}$", fontsize="large")

        fig.suptitle("Prediction", fontsize="xx-large")
        self._format_figure(fig, axes)

        return fig

    # Persistence -------------------------------------------------------------
    def save(self, savefile: str, overwrite: bool = False) -> None:
        """Save the plotting data in HDF5 format."""
        with opinf.utils.hdf5_savehandle(savefile, overwrite) as hf:
            for attr in (
                "sampling_time_domain",
                "training_time_domain",
                "prediction_time_domain",
                "snapshots_compressed",
                "true_states_compressed",
                "gp_means",
                "gp_stds",
                "draws_compressed",
            ):
                hf.create_dataset(attr, data=getattr(self, attr))

    @classmethod
    def load(cls, loadfile: str):
        """Load plotting data from an HDF5 file."""
        data = {}
        with opinf.utils.hdf5_loadhandle(loadfile) as hf:
            for attr in (
                "sampling_time_domain",
                "training_time_domain",
                "prediction_time_domain",
                "snapshots_compressed",
                "true_states_compressed",
                "gp_means",
                "gp_stds",
                "draws_compressed",
            ):
                data[attr] = hf[attr][:]
        return cls(**data)


@dataclasses.dataclass
class StatePlotter(baseplots._BasePlotter):
    """Generate plots in the original state space.

    Parameters
    ----------
    sample_time_domains : (num_samples,) ndarray
        Time domain over which noisy snapshots are observed.
    training_time_domain : (num_regression_points,) ndarray
        Time domain for the GP estimates used in the OpInf regression.
    prediction_time_domain : (k,) ndarray
        Time domain for true states and ROM predictions.
    spatial_domain : (nx,) ndarray
        Spatial domain (one-dimensional) over which to plot the results.
    num_variables : int
        Number of state variables, i.e., n = nx * num_variables.
    snapshots : (n, num_samples) ndarray
        Noisy compressed snapshots observed over ``sample_time_domain``.
    true_states : (n, k) ndarrays
        True (non-noised) states over the ``prediction_time_domain``.
    true_states_projected : (n, k) ndarray
        Projection of the true states over the ``prediction_time_domain``.
    draws : list of num_draws (n, k) ndarrays
        Collection of draws over ``prediction_time_domain``.
    numspatialpoints : int
        Number of spatial locations at which to plot the solution.
    """

    # Properties --------------------------------------------------------------
    spatial_domain: np.ndarray
    num_variables: int
    snapshots: np.ndarray
    true_states: np.ndarray
    true_states_projected: np.ndarray
    draws: list
    numspatialpoints: int = 8

    def __post_init__(self):
        """Downsample the data to select spatial locations."""
        if (periodic := self.numspatialpoints) == -1:
            self.numspatialpoints = 4
        if (N := self.snapshots.shape[0]) == (
            self.num_variables * self.numspatialpoints
        ):
            return self

        if periodic:
            mask = np.linspace(
                0,
                N // self.num_variables,
                self.numspatialpoints + 1,
                dtype=int,
            )[:-1]
        else:
            mask = np.linspace(
                0,
                (N // self.num_variables) - 1,
                self.numspatialpoints,
                dtype=int,
            )

        def downsample(Q):
            variables = np.split(Q, self.num_variables, axis=0)
            return np.concatenate([v[mask, :] for v in variables])

        self.spatial_domain = self.spatial_domain[mask]
        for attr in (
            "snapshots",
            "true_states",
            "true_states_projected",
        ):
            setattr(self, attr, downsample(getattr(self, attr)))
        self.draws = [downsample(draw) for draw in self.draws]

    # Utilities ---------------------------------------------------------------
    def new_figure(self, figsize=None):
        """Create a new figure and subplot axes."""
        if figsize is None:
            figsize = (12, self.numspatialpoints)
        rows = self.numspatialpoints // 2
        return plt.subplots(rows, 2, figsize=figsize, sharex=True)

    def _getvar(self, state, idx):
        return np.split(state, self.num_variables, axis=0)[idx]

    # Main routines -----------------------------------------------------------
    def plot_posterior(
        self,
        truth: bool = True,
        projected: bool = True,
        fulldomain: bool = True,
        individual: bool = False,
    ):
        """Plot the truth and its projection, the sparse/noisy data, and the
        ROM predictions (as individual draws) in the original state space.
        One figure is created per spatial location.

        Parameters
        ----------
        truth : bool
            If ``True`` (default), plot the true states.
            If ``False``, do not plot the true states.
        projected : bool
            If ``True`` (default), plot the projection of the true states.
            If ``False``, do not plot the projection of the true states.
        fulldomain : bool
            If ``True`` (default), plot the true and/or projected states over
            the full prediction domain.
            If ``False``, plot the true and/or projected states over only the
            training domain.
        individual : bool
            If ``True``, plot each of the draws individually.
            If ``False`` (default), plot the interquartile range of the draws.
        """
        end = None if fulldomain else self.end_train_index
        plotdraws = self._plot_draws if individual else self._plot_percentiles
        figures = []

        for d in range(self.num_variables):
            fig, axes = self.new_figure()
            for i, ax in enumerate(axes.flat):
                self._plot_data(
                    ax,
                    self.sampling_time_domain,
                    self._getvar(self.snapshots, d)[i],
                )
                ax.axvline(self.training_time_domain[-1], lw=1, color="black")

                if truth:
                    self._plot_truth(
                        ax,
                        self.prediction_time_domain[:end],
                        self._getvar(self.true_states, d)[i, :end],
                    )

                if projected:
                    self._plot_truth_projected(
                        ax,
                        self.prediction_time_domain[:end],
                        self._getvar(self.true_states_projected, d)[i, :end],
                    )

                draws = [self._getvar(draw, d)[i] for draw in self.draws]
                plotdraws(ax, self.prediction_time_domain, draws)
                self._plot_samplemean(ax, self.prediction_time_domain, draws)

                ax.set_title(
                    rf"$x = {self.spatial_domain[i]:.2f}$",
                    fontsize="large",
                )

            state = "q" if self.num_variables == 1 else f"q_{d+1}"
            fig.supylabel(f"${state}(x,t)$", fontsize="large")
            fig.suptitle("Prediction", fontsize="xx-large")
            self._format_figure(fig, axes)
            figures.append(fig)

        return figures

    # Persistence -------------------------------------------------------------
    def save(self, savefile: str, overwrite: bool = False) -> None:
        """Save the plotting data in HDF5 format."""
        with opinf.utils.hdf5_savehandle(savefile, overwrite) as hf:
            for attr in (
                "sampling_time_domain",
                "training_time_domain",
                "prediction_time_domain",
                "spatial_domain",
                "snapshots",
                "true_states",
                "true_states_projected",
                "draws",
            ):
                hf.create_dataset(attr, data=getattr(self, attr))
            for attr in "num_variables", "numspatialpoints":
                hf.create_dataset(attr, data=[getattr(self, attr)])

    @classmethod
    def load(cls, loadfile: str):
        """Load plotting data from an HDF5 file."""
        data = {}
        with opinf.utils.hdf5_loadhandle(loadfile) as hf:
            for attr in (
                "sampling_time_domain",
                "training_time_domain",
                "prediction_time_domain",
                "spatial_domain",
                "snapshots",
                "true_states",
                "true_states_projected",
                "draws",
            ):
                data[attr] = hf[attr][:]
            for attr in "num_variables", "numspatialpoints":
                data[attr] = int(hf[attr][0])
        return cls(**data)
