# baseplots.py
"""Matplotlib customization and base class for plotters."""

import abc
import dataclasses
import numpy as np
import matplotlib.pyplot as plt


# Matplotlib plot customization ===============================================
COLORS = {
    "data": "black",
    "true": "#666666",  # Gray
    "proj": "#823f00",  # Brown
    "draw": "#00add0",  # Light blue
    "mean": "#ac00dc",  # Purple
    "GPcs": "#ff7f0e",  # Orange
}

plt.rc("axes", linewidth=0.5)
plt.rc("axes.spines", top=False, right=False)
plt.rc("figure", dpi=300, figsize=(12, 8))
plt.rc("font", family="serif", size=16)
plt.rc("legend", edgecolor="none", frameon=False)
plt.rc("text", usetex=True)


@dataclasses.dataclass
class _BasePlotter(abc.ABC):
    """Abstract base class for plotters.

    Parameters
    ----------
    sampling_time_domain : (num_samples,) ndarray
        Time domain over which noisy snapshots are observed.
    training_time_domain : (num_regression_points,) ndarray
        Time domain for the GP estimates used in the OpInf regression.
    prediction_time_domain : (k,) ndarray
        Time domain for true states and model predictions.
    """

    # Properties --------------------------------------------------------------
    sampling_time_domain: np.ndarray
    training_time_domain: np.ndarray
    prediction_time_domain: np.ndarray

    @property
    def end_train_index(self) -> int:
        """Index of the full time domain where the training regime ends."""
        endtrain = self.training_time_domain[-1]
        return np.argmin(self.prediction_time_domain < endtrain) + 1

    # Utilities ---------------------------------------------------------------
    @staticmethod
    def _format_figure(fig: plt.Figure, axes: plt.Axes, legend=True):
        """Apply labels, legends, and alignment."""
        if axes.ndim == 1:
            axes = axes.reshape((-1, 1))

        # Format axis labels.
        for ax in axes[-1, :]:
            ax.set_xlabel("$t$")
        for j in range(axes.shape[1]):
            fig.align_ylabels(axes[:, j])

        # Make legend centered below the subplots.
        if legend:
            fig.tight_layout(rect=[0, 0.075, 1, 1])
            leg = axes[0, -1].legend(
                ncol=len(axes[0, -1].get_legend_handles_labels()[1]),
                loc="lower center",
                fontsize="large",
                bbox_to_anchor=(0.5, 0),
                bbox_transform=fig.transFigure,
            )
            for line in leg.get_lines():
                line.set_linewidth(5)
                line.set_markersize(14)
                line.set_alpha(1)

        return fig, (axes[:, 0] if axes.shape[1] == 1 else axes)

    @staticmethod
    def _format_params(params):
        """Input parameters -> readable string."""
        return "(" + ", ".join([f"{p:.2f}" for p in params]) + ")"

    # Line plotters -----------------------------------------------------------
    @staticmethod
    def _plot_truth(ax, t, truth, **kwargs):
        """Plot true states, either in reduced space or at a spatial point."""
        kws = dict(
            linestyle="-",
            linewidth=1,
            color=COLORS["true"],
            label="Truth",
            zorder=2,
        )
        kws.update(kwargs)
        ax.plot(t, truth, **kws)

    @staticmethod
    def _plot_truth_projected(ax, t, truth_projected, **kwargs):
        """Plot the projection of true states at a spatial point."""
        kws = dict(
            linewidth=1,
            linestyle="-.",
            color=COLORS["proj"],
            label="Truth projected",
            zorder=2,
        )
        kws.update(kwargs)
        ax.plot(t, truth_projected, **kws)

    @staticmethod
    def _plot_data(ax, t, data, **kwargs):
        """Plot noisy data, either in reduced space or at a spatial point."""
        kws = dict(
            linestyle="",
            marker="*",
            markersize=4,
            markerfacecolor=COLORS["data"],
            markeredgewidth=0,
            label="Data",
            zorder=4,
        )
        kws.update(kwargs)

        ax.plot(t, data, **kws)

    @staticmethod
    def _plot_gp(ax, t, mean, stdev, width, **kwargs):
        """Plot the mean and variance of a GP after regression."""
        spread = width * stdev
        low, high = mean - spread, mean + spread

        kws = dict(
            color=COLORS["GPcs"],
            linestyle="--",
            linewidth=1,
            alpha=0.3,
            zorder=3,
        )
        kws.update(kwargs)
        label = rf"GP $\mu \pm {f'{width:d}' if width != 1 else ''}\sigma$"

        ax.plot(
            t,
            mean,
            ls=kws["linestyle"],
            lw=kws["linewidth"],
            color=kws["color"],
            label=label,
            zorder=kws["zorder"],
        )
        ax.fill_between(
            t,
            low,
            high,
            lw=0,
            color=kws["color"],
            alpha=kws["alpha"],
        )

    @staticmethod
    def _plot_samplemean(ax, t, draws, **kwargs):
        """Plot the sample mean of a time-dependent ensemble."""
        kws = dict(
            linewidth=1,
            linestyle="--",
            color=COLORS["mean"],
            label="Sample mean",
            zorder=3,
        )
        kws.update(kwargs)
        ax.plot(t, np.mean(draws, axis=0), **kws)

    @staticmethod
    def _plot_draws(ax, t, draws, **kwargs):
        """Plot a time-dependent ensemble as individual draws."""
        kws = dict(lw=0.5, alpha=0.15, color=COLORS["draw"], zorder=0.5)
        kws.update(kwargs)
        ax.plot(t, draws[0], label="Posterior draws", **kws)
        for draw in draws[1:]:
            ax.plot(t, draw, **kws)

    @staticmethod
    def _plot_percentiles(ax, t, draws, ipr=95, **kwargs):
        """Shade in the ``ipr``% interquartile range of the draws."""
        low = (100 - ipr) / 2
        high = 100 - low
        percentiles = np.percentile(draws, [low, high], axis=0)

        kws = dict(
            lw=0,
            color=COLORS["draw"],
            alpha=0.4,
            label=rf"Sampled ROM predictions ${ipr}\%$ IQR",
        )
        kws.update(**kwargs)
        ax.fill_between(t, percentiles[0], percentiles[1], **kws)

    # Main methods (abstract) -------------------------------------------------
    @abc.abstractmethod
    def new_figure(self, figsize=None):
        """Create a new figure and subplot axes."""
        raise NotImplementedError

    @abc.abstractmethod
    def plot_posterior(
        self,
        truth: bool = True,
        fulldomain: bool = True,
        individual: bool = True,
    ):
        """Visualize the model posterior.

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
        raise NotImplementedError
