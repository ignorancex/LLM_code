# step4_plot.py
"""Make plots for GP-BayesOpInf ODE examples."""

import dataclasses
import numpy as np
import matplotlib.pyplot as plt

import config
import baseplots

import opinf


@dataclasses.dataclass
class ODEPlotter(baseplots._BasePlotter):
    """Generate plots in the ODE state space.

    Parameters
    ----------
    sampling_time_domains : list of num_states (num_samples,) ndarrays
        Time domain over which noisy snapshots are observed.
    training_time_domain : (num_regression_points,) ndarray
        Time domain for the GP estimates used in the OpInf regression.
    prediction_time_domain : (k,) ndarray
        Time domain for true states and model predictions.
    snapshots : (num_states, num_samples) ndarray
        Noisy snapshots observed over ``sample_time_domain``.
    true_states : (num_states, k) ndarray
        True (non-noised) states over the ``prediction_time_domain``.
    gpmeans : list of num_states (num_regression_points,) ndarrays
        Gaussian process means over ``training_time_domain``.
    gpstds : list of num_states (num_regression_points,) ndarrays
        Gaussian process standard deviations over ``training_time_domain``.
    draws : list of num_draws (num_states, k) ndarrays
        Collection of draws over ``prediction_time_domain``.
    labels : list of str
        Labels for the state variables.
    """

    # Properties --------------------------------------------------------------
    snapshots: np.ndarray
    true_states: np.ndarray
    gp_means: np.ndarray
    gp_stds: np.ndarray
    draws: list
    labels: list

    @property
    def num_states(self) -> int:
        """Number of state variables."""
        return len(self.snapshots)

    # Utilities ---------------------------------------------------------------
    def new_figure(self, figsize=None):
        """Create a new figure and subplot axes."""
        if figsize is None:
            figsize = (6, 2 * self.num_states)
        return plt.subplots(
            self.num_states,
            1,
            figsize=figsize,
            sharex=True,
            # sharey=True,
        )

    @property
    def end_train_index(self) -> int:
        """Index of the full time domain where the training regime ends."""
        return (
            np.argmin(
                self.prediction_time_domain < self.sampling_time_domain[0][-1]
            )
            + 1
        )

    # Main routines -----------------------------------------------------------
    def plot_gp_training_fit(self, width=3):
        """Plot the truth, the sparse / noisy data, and the GP fit to the data.

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
                self.true_states[i, :end],
            )
            self._plot_data(
                ax,
                self.sampling_time_domain[i],
                self.snapshots[i],
            )
            self._plot_gp(
                ax,
                self.training_time_domain,
                self.gp_means[i],
                self.gp_stds[i],
                width=width,
            )
            ax.set_ylabel(self.labels[i], fontsize="large")

        fig.suptitle("GP fit", fontsize="xx-large")
        self._format_figure(fig, axes)

        return fig

    def plot_posterior(
        self,
        truth: bool = True,
        fulldomain: bool = True,
        individual: bool = False,
    ):
        """Plot the truth, sparse/noisy data, and model predictions.

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
                    self.true_states[i, :end],
                )

            self._plot_data(
                ax,
                self.sampling_time_domain[i],
                self.snapshots[i],
            )

            draw_subset = [draw[i] for draw in self.draws]
            plotdraws(ax, self.prediction_time_domain, draw_subset)
            self._plot_samplemean(ax, self.prediction_time_domain, draw_subset)

            ax.axvline(self.training_time_domain[-1], lw=1, color="black")
            ax.set_ylabel(self.labels[i], fontsize="large")

        fig.suptitle("Prediction", fontsize="xx-large")
        self._format_figure(fig, axes)

        return fig

    def plot_posterior_newICs(
        self,
        draws,
        truth=None,
        individual: bool = False,
    ):
        """Plot the truth, sparse/noisy data, and model predictions for
        a new set of data.

        Parameters
        ----------
        draws : list of (num_draws, k) ndarrays
            Posterior model predictions.
        truth : (num_states, k) ndarray
            True solution that the draws are approximating, measured over
            the ``prediction_time_domain``.
        individual : bool
            If ``True``, plot each of the draws individually.
            If ``False`` (default), plot the interquartile range of the draws.
        """
        plotdraws = self._plot_draws if individual else self._plot_percentiles
        fig, axes = self.new_figure()

        for i, ax in enumerate(axes.flat):
            if truth is not None:
                self._plot_truth(
                    ax,
                    self.prediction_time_domain,
                    truth[i],
                )
            draw_subset = [draw[i] for draw in draws]
            plotdraws(ax, self.prediction_time_domain, draw_subset)
            self._plot_samplemean(ax, self.prediction_time_domain, draw_subset)
            ax.set_ylabel(self.labels[i], fontsize="large")

        fig.suptitle(
            "Prediction (new initial conditions)",
            fontsize="xx-large",
        )
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
                "snapshots",
                "true_states",
                "gp_means",
                "gp_stds",
                "draws",
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
                "snapshots",
                "true_states",
                "gp_means",
                "gp_stds",
                "draws",
            ):
                data[attr] = hf[attr][:]
        return cls(**data, labels=config.Model.LABELS)
