# step1_generate_data.py
"""Generate (noisy) data for numerical experiments."""

__all__ = [
    "trajectory",
]

import numpy as np

import opinf

import config


def trajectory(
    training_span: tuple[float, float],
    num_samples: int,
    noiselevel: float = 0.0,
):
    """Get sparse, noisy data for a single trajectory.

    Parameters
    ----------
    training_span : (float, float)
        Time domain over which to sample solution data.
    num_samples : int > 0
        Number of snapshots to sample.
    noiselevel : float >= 0
        Percentage of noise applied to the sampled snapshots.

    Returns
    -------
    model
    full_time_domain
    true_states
    time_domain
    snapshots
    """
    with opinf.utils.TimedBlock("generating training data"):
        # Initialize and solve the truth model over the full domain
        model = config.FullOrderModel()
        true_states = model.solve(
            config.initial_conditions,
            config.time_domain,
        )

        # Uniformly sample from the training span --> training time domain.
        time_domain = np.sort(
            np.random.uniform(
                training_span[0],
                training_span[1],
                size=num_samples,
            )
        )
        time_domain[0] = training_span[0]
        time_domain[-1] = training_span[1]

        # Get noisy snapshots over the training time domain.
        snapshots = model.noise(
            model.solve(config.initial_conditions, time_domain),
            noiselevel,
        )

        return (
            model,
            config.time_domain,
            true_states,
            time_domain,
            snapshots,
        )
