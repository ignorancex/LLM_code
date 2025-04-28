# step1_generate_data.py
"""Generate (noisy) data for numerical experiments."""

__all__ = [
    "TrajectorySampler",
]

import numpy as np

import config


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
        If ``True`` (default), samples occur at the same times for all state
        variables.
        If ``False``, samples occur at different times for each state variable.
    integersonly : bool
        If ``True``, restrict sampling times to the integers.

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
        synced: bool = True,
        integersonly: bool = False,
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
        self.integersonly = integersonly

        self.synced = synced
        if synced:
            self.synced_sample_time_domain = self._sample_time_domain()

    def _sample_time_domain(self):
        """Draw uniformly from the training span to generate a time domain
        over which to observe noisy snapshots.
        """
        if self.integersonly:
            t = np.random.choice(
                int(self.training_span[1]),
                size=self.num_samples,
                replace=False,
            )
        else:
            t = np.random.uniform(
                self.training_span[0],
                self.training_span[1],
                size=self.num_samples,
            )

        times = np.sort(t)
        times[0] = self.training_span[0]
        times[-1] = self.training_span[1]
        return times

    def sample(self):
        """Get data for a single trajectory.

        Parameters
        ----------
        input_parameters : tuple
            Parameter values for the model input function,
            the sole argument for the model constructor.

        Returns
        -------
        model
            Model to use for parameter estimation.
        prediction_time_domain : (K,) ndarray
            Time domain for the prediction and the true states.
        true_states : (K,) ndarray
            Non-noisy states over ``config.time_domain``.
        sample_time_domain : (num_samples,) ndarray
            Time domain over which noisy snapshots are observed.
        snapshots : (state_dimension, num_samples) ndarray
            Noisy snapshots observed over ``sample_time_domain``.
        """
        X0 = config.initial_conditions
        t_predict = config.time_domain

        # Truth model.
        model = config.Model()
        true_states = model.solve(X0, t_predict)

        # Noisy data.
        if self.synced:
            # All state variables observed at the same times.
            t = self.synced_sample_time_domain
            snapshots = model.noise(model.solve(X0, t), self.noiselevel)
            sample_times = [t] * snapshots.shape[0]
        else:
            # Each state variable observed at different times.
            sample_times, snapshots = [], []
            for i in range(model.num_variables):
                t = self._sample_time_domain()
                noised = model.noise(model.solve(X0, t), self.noiselevel)
                snapshots.append(noised[i, :])
                sample_times.append(t)

        return model, t_predict, true_states, sample_times, snapshots
