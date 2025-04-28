# step1_generate_data.py
"""Generate (noisy) data for numerical experiments."""

__all__ = [
    "TrajectorySampler",
]

import numpy as np
import matplotlib.pyplot as plt

import config
import utils


class TrajectorySampler:
    """Get sparse, noisy data for a several trajectories.

    Parameters
    ----------
    training_span : (float, float)
        Time domain over which to sample solution data.
    num_samples : int > 0
        Number of snapshots to sample.
    noiselevel : 0 <= float <= 1
        Percentage of noise applied to the sampled snapshots.
    num_regression_points : int > 0
        Number of points at which to evaluate the GP state and derivative
        estimates.
    synced : bool
        If ``True``, data are sampled at the same times for all trajectories.
        If ``False`` (default), data from different trajectories are sampled
        at different times.

    Attributes
    ----------
    training_time_domain : (num_regression_points,) ndarray
        Time domain at which to evaluate the GP state and derivative estimates.
    prediction_time_domain : (k,) ndarray
        Time domain for ROM predictions and true state data, i.e.,
        ``config.time_domain``.
    """

    def __init__(
        self,
        training_span: tuple[float, float],
        num_samples: int,
        noiselevel: float,
        num_regression_points: int,
        synced: bool = False,
    ):
        """Set sampler configuration."""
        self.training_span = training_span
        self.num_samples = num_samples
        self.noiselevel = noiselevel

        # Equally spaced training time domain for GP estimates.
        self.training_time_domain = np.linspace(
            self.training_span[0],
            self.training_span[1],
            num_regression_points,
        )

        self.prediction_time_domain = config.time_domain

        self.synced = synced
        if synced:
            self.synced_sample_time_domain = self._sample_time_domain()

    def _sample_time_domain(self):
        """Draw uniformly from the training span to generate a time domain
        over which to observe noisy snapshots.
        """
        times = np.sort(
            np.random.uniform(
                self.training_span[0],
                self.training_span[1],
                size=self.num_samples,
            )
        )
        times[0] = self.training_span[0]
        times[-1] = self.training_span[1]
        return times

    def sample(self, input_parameters):
        """Get data for a single trajectory.

        Parameters
        ----------
        input_parameters : tuple
            Parameter values for the model input function,
            the sole argument for the model constructor.

        Returns
        -------
        true_states : (k,) ndarray
            Non-noisy states over ``config.time_domain``.
        sample_time_domain : (num_samples,) ndarray
            Time domain over which noisy snapshots are observed.
        snapshots : (state_dimension, num_samples) ndarray
            Noisy snapshots observed over ``sample_time_domain``.
        training_inputs : (num_regression_points,) ndarray
            Model inputs evaluated over the regression domain.
        """
        # Truth model.
        model = config.FullOrderModel(input_parameters)
        true_states = model.solve(
            config.initial_conditions,
            config.time_domain,
        )

        # Time domain over which to observe noisy data.
        if self.synced:
            sample_time_domain = self.synced_sample_time_domain
        else:
            sample_time_domain = self._sample_time_domain()

        # Noisy data.
        snapshots = model.noise(
            model.solve(config.initial_conditions, sample_time_domain),
            self.noiselevel,
        )

        # Inputs for the regression.
        input_func = config.input_func_factory(input_parameters)
        training_inputs = input_func(self.training_time_domain)

        return true_states, sample_time_domain, snapshots, training_inputs

    def multisample(self, input_parameter_set, plot: bool = False):
        """Get data for multiple trajectories.

        Parameters
        ----------
        input_parameter_set : list(tuple) of length L
            Collection of parameter values for the model input function.
        plot : bool
            If ``True`` and the ``input_parameter_set`` is two-dimensional,
            visualize the parameter set.

        Returns
        -------
        true_states : list of L (k,) ndarrays
            Non-noisy states over ``config.time_domain``.
        sample_time_domains : list of L (num_samples,) ndarrays
            Time domain over which noisy snapshots are observed.
        snapshots : list of L (state_dimension, num_samples) ndarrays
            Noisy snapshots observed over ``sample_time_domain``.
        training_inputs : list of L (num_regression_points,) ndarrays
            Model inputs evaluated over the regression domain.
        """
        states, sample_domains, snapshots, training_inputs = [], [], [], []

        if plot and len(input_parameter_set[0]) == 2:
            inputs = np.array(input_parameter_set)
            fig, ax = plt.subplots(1, 1, figsize=(9, 9))
            ax.plot(inputs[:, 0], inputs[:, 1], "k*")

            low, high = inputs.min(), inputs.max()
            mid = (high + low) / 2
            width = (high - low) * 0.6
            low, high = mid - width, mid + width
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)

            ax.set_xlabel(r"$a$")
            ax.set_ylabel(r"$b$")
            ax.set_aspect("equal")
            ax.set_title("Training trajectories")
            utils.save_figure(
                "trajectory_parameters.pdf",
                andopen=True,
                fig=fig,
            )

        for params in input_parameter_set:
            truth, sample_times, snaps, inputs = self.sample(params)

            states.append(truth)
            sample_domains.append(sample_times)
            snapshots.append(snaps)
            training_inputs.append(inputs)

        return states, sample_domains, snapshots, training_inputs
