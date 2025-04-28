# step4_plot.py
"""Make plots for GP-BayesOpInf PDE examples with multiple trajectories."""

__all__ = [
    "ReducedPlotter",
    "StatePlotter",
]

import dataclasses
import numpy as np
import matplotlib.pyplot as plt

import baseplots

import opinf


@dataclasses.dataclass
class _BaseMultiPlotter(baseplots._BasePlotter):
    """Add trajectory_parameters to the attributes."""

    trajectory_parameters: list

    @property
    def num_trajectories(self) -> int:
        """Number of training trajectories."""
        return len(self.trajectory_parameters)


@dataclasses.dataclass
class ReducedPlotter(_BaseMultiPlotter):
    """Generate plots in the reduced space.

    Parameters
    ----------
    sample_time_domains : list of L (num_samples,) ndarrays
        Time domain over which noisy snapshots are observed.
    training_time_domain : (num_regression_points,) ndarray
        Time domain for the GP estimates used in the OpInf regression.
    prediction_time_domain : (k,) ndarray
        Time domain for true states and ROM predictions.
    trajectory_parameters : list of L tuple(float)
        Parameters defining each trajectory.
    snapshots_compressed : list of L (r, num_samples) ndarrays
        Noisy compressed snapshots observed over ``sample_time_domain``.
    true_states_compressed : list of L (r, k) ndarrays
        True (non-noised) states over the ``prediction_time_domain``,
        indexed by trajectory, then mode, then time step.
    draws_compressed : list of L lists of num_draws (num_modes, k) ndarrays
        Collection of compressed draws for each trajectory.
    """

    # Properties --------------------------------------------------------------
    snapshots_compressed: list[np.ndarray]
    true_states_compressed: list[np.ndarray]
    gp_means: list[np.ndarray]
    gp_stds: list[np.ndarray]
    draws_compressed: list
    max_modes: int = 8

    @property
    def num_modes(self) -> int:
        """Number of basis functions / reduced state dimension."""
        return self.snapshots_compressed[0].shape[0]

    # Main routines -----------------------------------------------------------
    def new_figure(self, figsize=None):
        """Create a new figure and subplot axes."""
        rows, cols = (self.num_trajectories, 1)
        if figsize is None:
            figsize = (6, 2 * self.num_trajectories)
        if self.num_trajectories == 15:
            figsize = (9, 12)
            rows, cols = 5, 3

        return plt.subplots(rows, cols, figsize=figsize, sharex=True)

    def plot_gp_training_fit(self, width=3):
        """Plot the truth, the sparse / noisy / compressed data, and the GP fit
        to the data.

        Parameters
        ----------
        gps : list of L lists of num_modes GPs
            Trained GPs.
        width : float
            Plot mean ± (``width`` * standard_deviation).
        """
        figures = []
        end = self.end_train_index
        for i in range(self.num_modes):
            fig, axes = self.new_figure()
            for ell, ax in enumerate(axes.flat):
                self._plot_truth(
                    ax,
                    self.prediction_time_domain[:end],
                    self.true_states_compressed[ell][i, :end],
                )
                self._plot_data(
                    ax,
                    self.sampling_time_domain[ell],
                    self.snapshots_compressed[ell][i],
                )
                self._plot_gp(
                    ax,
                    self.training_time_domain,
                    self.gp_means[ell][i],
                    self.gp_stds[ell][i],
                    width=width,
                )
                # ps = self._format_params(self.trajectory_parameters[ell])
                ax.set_title(
                    rf"trajectory {ell + 1:d}",  # params={ps},
                    fontsize="large",
                )
            fig.suptitle(rf"GP fit, $r = {i + 1:d}$", fontsize="xx-large")

            self._format_figure(fig, axes)
            figures.append(fig)

        return figures

    def plot_posterior(
        self,
        truth: bool = True,
        fulldomain: bool = True,
        individual: bool = False,
    ):
        """Plot the truth, data, and ROM predictions.

        Parameters
        ----------
        truth : bool
            If ``True`` (default), plot the true states.
            If ``False``, do not plot the true states.
        fulldomain : bool
            If ``True`` (default), plot the true states over the full
            prediction domain.
            If ``False``, plot the true states only over the training domain.
        individual : bool
            If ``True``, plot each of the draws individually.
            If ``False`` (default), plot the interquartile range of the draws.
        """
        end = None if fulldomain else self.end_train_index
        plotdraws = self._plot_draws if individual else self._plot_percentiles
        figures = []

        for i in range(self.num_modes):
            fig, axes = self.new_figure()
            for ell, ax in enumerate(axes.flat):
                if truth:
                    self._plot_truth(
                        ax,
                        self.prediction_time_domain[:end],
                        self.true_states_compressed[ell][i, :end],
                    )
                self._plot_data(
                    ax,
                    self.sampling_time_domain[ell],
                    self.snapshots_compressed[ell][i],
                )

                draws = [draw[i] for draw in self.draws_compressed[ell]]
                plotdraws(ax, self.prediction_time_domain, draws)
                self._plot_samplemean(ax, self.prediction_time_domain, draws)

                # ps = self._format_params(self.trajectory_parameters[ell])
                ax.set_title(
                    rf"trajectory {ell + 1:d}",  # , params = {ps}")
                    fontsize="large",
                )
                ax.axvline(self.training_time_domain[-1], lw=1, color="black")
            fig.suptitle(rf"Prediction $r = {i + 1:d}$", fontsize="xx-large")

            self._format_figure(fig, axes)
            figures.append(fig)

        return figures

    def plot_posterior_newparams(self, draws, truth, individual: bool = False):
        """Plot the truth and the ROM predictions mapped to the original state
        space for a single set of parameter values.

        Parameters
        ----------
        draws : list of num_draws (n, k) ndarrays
            Draws for the new trajectory.
        truth : (n, k) ndarray
            True trajectory.
        individual : bool
            If ``True``, plot the each of the draws individually.
            If ``False`` (default), plot the interquartile range of the draws.
        """
        fig, axes = plt.subplots(
            self.num_modes,
            1,
            figsize=(6, 2 * self.num_modes),
            sharex=True,
        )
        drawplot = self._plot_draws if individual else self._plot_percentiles

        for i, ax in enumerate(axes.flat):
            self._plot_truth(ax, self.prediction_time_domain, truth[i])

            draw_subset = [draw[i] for draw in draws]
            drawplot(ax, self.prediction_time_domain, draw_subset)
            self._plot_samplemean(ax, self.prediction_time_domain, draw_subset)

            ax.axvline(self.training_time_domain[-1], lw=1, color="black")
            ax.set_title(rf"$r = {i + 1}$", fontsize="large")

        fig.suptitle("New trajectory", fontsize="xx-large")
        self._format_figure(fig, axes)

        return fig

    # Persistence -------------------------------------------------------------
    def save(self, savefile: str, overwrite: bool = False) -> None:
        """Save the plotting data in HDF5 format."""
        with opinf.utils.hdf5_savehandle(savefile, overwrite) as hf:
            for attr in (
                "trajectory_parameters",
                "sampling_time_domain",
                "training_time_domain",
                "prediction_time_domain",
                "snapshots_compressed",
                "true_states_compressed",
                "gp_means",
                "gp_stds",
            ):
                hf.create_dataset(attr, data=getattr(self, attr))
            for k, draw in enumerate(self.draws_compressed):
                hf.create_dataset(f"draw_{k+1:0>3d}", data=draw)
            hf.create_dataset("ndraws", data=[len(self.draws_compressed)])

    @classmethod
    def load(cls, loadfile: str):
        """Load plotting data from an HDF5 file."""
        data = {}
        with opinf.utils.hdf5_loadhandle(loadfile) as hf:
            for attr in (
                "trajectory_parameters",
                "sampling_time_domain",
                "training_time_domain",
                "prediction_time_domain",
                "snapshots_compressed",
                "true_states_compressed",
                "gp_means",
                "gp_stds",
            ):
                data[attr] = hf[attr][:]
            data["draws_compressed"] = [
                hf[f"draw_{k+1:>03d}"][:] for k in range(int(hf["ndraws"][0]))
            ]

        return cls(**data)


@dataclasses.dataclass
class StatePlotter(_BaseMultiPlotter):
    """Generate plots in the original state space.

    Parameters
    ----------
    sample_time_domains : list of L (num_samples,) ndarrays
        Time domain over which noisy snapshots are observed.
    training_time_domain : (num_regression_points,) ndarray
        Time domain for the GP estimates used in the OpInf regression.
    prediction_time_domain : (k,) ndarray
        Time domain for true states and ROM predictions.
    trajectory_parameters : list of L tuple(float)
        Parameters defining each trajectory.
    spatial_domain : (nx,) ndarray
        Spatial domain (one-dimensional) over which to plot the results.
    num_variables : int
        Number of state variables, i.e., n = nx * num_variables.
    snapshots : list of L (n, num_samples) ndarrays
        Noisy compressed snapshots observed over ``sample_time_domain``.
    true_states : list of L (n, k) ndarrays
        True (non-noised) states over the ``prediction_time_domain``,
        indexed by trajectory, then spatial location, then time step.
    true_states_projected : list of L (n, k) ndarrays
        Projection of the true states over the ``prediction_time_domain``.
    numspatialpoints : int
        Number of spatial locations at which to plot the solution.
    """

    # Properties --------------------------------------------------------------
    spatial_domain: np.ndarray
    num_variables: int
    snapshots: list[np.ndarray]
    true_states: list[np.ndarray]
    true_states_projected: list[np.ndarray]
    draws: list[list]
    numspatialpoints: int = 8

    def __post_init__(self):
        """Downsample the data to select spatial locations."""
        if (N := self.snapshots[0].shape[0]) == (
            self.num_variables * self.numspatialpoints
        ):
            return self

        if self.numspatialpoints == -1:
            xx = [1 / 8, 1 / 4, 1 / 2, 3 / 4, 7 / 8]
            self.numspatialpoints = len(xx)
            mask = np.array(
                [np.argmin(np.abs(self.spatial_domain - x)) for x in xx]
            )
        else:
            nx = N // self.num_variables
            # Without BCs
            # mask = np.linspace(
            #     0,
            #     nx - 1,
            #     self.numspatialpoints + 2,
            #     dtype=int,
            # )[1:-1]
            # With BCs
            mask = np.linspace(0, nx - 1, self.numspatialpoints, dtype=int)

        def downsample(Q):
            variables = np.split(Q, self.num_variables, axis=0)
            return np.concatenate([v[mask, :] for v in variables])

        self.spatial_domain = self.spatial_domain[mask]
        for attr in (
            "snapshots",
            "true_states",
            "true_states_projected",
        ):
            setattr(self, attr, [downsample(Q) for Q in getattr(self, attr)])
        self.draws = [
            [downsample(draw) for draw in trajectory]
            for trajectory in self.draws
        ]

    # Utilities ---------------------------------------------------------------
    def new_figure(self, figsize=None):
        """Create a new figure and subplot axes."""
        if figsize is None:
            figsize = (12, self.numspatialpoints)
        return plt.subplots(
            self.numspatialpoints // 2,
            2,
            figsize=figsize,
            sharex=True,
            # sharey=True,
        )

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
        ROM predictions mapped to the original state space.
        One figure per trajectory per state variable.

        Parameters
        ----------
        draws : list of L lists of num_draws (n, k) ndarrays
            Collection of draws for each trajectory.
        truth : bool
            If ``True`` (default), plot the true states.
            If ``False``, do not plot the true states.
        projected : bool
            If ``True`` (default), plot the projection of the true states.
            If ``False``, do not plot the projection of the true states.
        fulldomain : bool
            If ``True`` (default), plot the true states over the full
            prediction domain.
            If ``False``, plot the true states only over the training domain.
            The projected truth is always drawn over the prediction domain.
        individual : bool
            If ``True``, plot each of the draws individually.
            If ``False`` (default), plot the interquartile range of the draws.
        """
        end = None if fulldomain else self.end_train_index
        plotdraws = self._plot_draws if individual else self._plot_percentiles
        all_figures = []

        for d in range(self.num_variables):
            figures = []
            for ell, params in enumerate(self.trajectory_parameters):
                # ps = self._format_params(params)

                fig, axes = self.new_figure()
                for i, ax in enumerate(axes.flat):
                    self._plot_data(
                        ax,
                        self.sampling_time_domain[ell],
                        self._getvar(self.snapshots[ell], d)[i],
                    )
                    ax.axvline(
                        self.training_time_domain[-1], lw=1, color="black"
                    )

                    if truth:
                        self._plot_truth(
                            ax,
                            self.prediction_time_domain[:end],
                            self._getvar(self.true_states[ell], d)[i, :end],
                        )

                    if projected:
                        self._plot_truth_projected(
                            ax,
                            self.prediction_time_domain[:end],
                            self._getvar(
                                self.true_states_projected[ell],
                                d,
                            )[i, :end],
                        )

                    draws = [
                        self._getvar(draw, d)[i] for draw in self.draws[ell]
                    ]
                    plotdraws(ax, self.prediction_time_domain, draws)
                    self._plot_samplemean(
                        ax, self.prediction_time_domain, draws
                    )

                    ax.set_title(
                        rf"$x = {self.spatial_domain[i]:.2f}$",
                        fontsize="large",
                    )

                state = "q" if self.num_variables == 1 else f"q_{d+1}"
                fig.supylabel(f"${state}(x,t)$", fontsize="large")
                fig.suptitle(
                    rf"Trajectory {ell + 1:d}",  # , params = {ps}")
                    fontsize="xx-large",
                )
                self._format_figure(fig, axes)
                figures.append(fig)
            all_figures.append(figures)

        return all_figures

    def plot_posterior_newparams(
        self,
        draws,
        truth,
        spatial_domain=None,
        individual: bool = False,
    ):
        """Plot the truth and the ROM predictions mapped to the original state
        space for a single set of parameter values.

        Parameters
        ----------
        draws : list of num_draws (n, k) ndarrays
            Draws for the new trajectory.
        truth : (n, k) ndarray
            True trajectory.
        spatial_domain : (n,) ndarray
            Spatial domain corresponding to the states.
        individual : bool
            If ``True``, plot each of the draws individually.
            If ``False`` (default), plot the interquartile range of the draws.
        """
        # First, downsample to spatial indices to plot.
        if truth.shape[0] != self.spatial_domain.size:
            if spatial_domain is None:
                raise ValueError("spatial_domain required")
            mask = [
                np.argmin(np.abs(spatial_domain - x))
                for x in self.spatial_domain
            ]
            truth = truth[mask, :]
            draws = [draw[mask, :] for draw in draws]

        plotdraws = self._plot_draws if individual else self._plot_percentiles
        figures = []

        for d in range(self.num_variables):
            fig, axes = self.new_figure()
            for i, ax in enumerate(axes.flat):
                ax.axvline(self.training_time_domain[-1], lw=1, color="black")

                self._plot_truth(
                    ax,
                    self.prediction_time_domain,
                    self._getvar(truth, d)[i],
                )

                draws_ = [self._getvar(draw, d)[i] for draw in draws]
                plotdraws(ax, self.prediction_time_domain, draws_)
                self._plot_samplemean(ax, self.prediction_time_domain, draws_)

                ax.set_title(
                    rf"$x = {self.spatial_domain[i]:.2f}$",
                    fontsize="large",
                )

            fig.supylabel(r"$q(x,t)$", fontsize="large")
            fig.suptitle("New trajectory", fontsize="xx-large")
            self._format_figure(fig, axes)
            figures.append(fig)

        return figures

    # Persistence -------------------------------------------------------------
    def save(self, savefile: str, overwrite: bool = False) -> None:
        """Save the plotting data in HDF5 format."""
        with opinf.utils.hdf5_savehandle(savefile, overwrite) as hf:
            for attr in (
                "trajectory_parameters",
                "sampling_time_domain",
                "training_time_domain",
                "prediction_time_domain",
                "spatial_domain",
                "snapshots",
                "true_states",
                "true_states_projected",
            ):
                hf.create_dataset(attr, data=getattr(self, attr))
            for attr in "num_variables", "numspatialpoints":
                hf.create_dataset(attr, data=[getattr(self, attr)])
            hf.create_dataset(
                "ndraws",
                data=[len(draws) for draws in self.draws],
            )
            for ell in range(self.num_trajectories):
                for k, draw in enumerate(self.draws[ell]):
                    hf.create_dataset(f"draw_{ell:0>2d}-{k:0>3d}", data=draw)

    @classmethod
    def load(cls, loadfile: str):
        """Load plotting data from an HDF5 file."""
        data = {}
        with opinf.utils.hdf5_loadhandle(loadfile) as hf:
            for attr in (
                "trajectory_parameters",
                "sampling_time_domain",
                "training_time_domain",
                "prediction_time_domain",
                "spatial_domain",
                "snapshots",
                "true_states",
                "true_states_projected",
            ):
                data[attr] = hf[attr][:]
            data["num_variables"] = int(hf["num_variables"][0])
            data["numspatialpoints"] = int(hf["numspatialpoints"][0])
            ndraws = hf["ndraws"][:]
            data["draws"] = [
                [
                    hf[f"draw_{ell:0>2d}-{k:0>3d}"][:]
                    for k in range(ndraws[ell])
                ]
                for ell in range(len(data["trajectory_parameters"]))
            ]

        return cls(**data)
