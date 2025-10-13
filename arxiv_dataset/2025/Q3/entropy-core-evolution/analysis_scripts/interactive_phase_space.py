'''
interactive_phase_space.py

Visualize and interactively explore the evolution of gas phase space data
as a function of redshift. Use the slider to step through snapshots and
see how the phase space distribution changes over time.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, Button


def load_phase_space_data(parent_dir: str):
    """
    Load the phase space histogram and redshift history from NumPy files.

    Parameters
    ----------
    parent_dir : str
        Directory where the data files are located.

    Returns
    -------
    phase_space_data : np.ndarray
        3D array of shape (num_snapshots, num_density_bins, num_temperature_bins).
    redshift_history : np.ndarray
        1D array of redshift values corresponding to each snapshot.
    """
    phase_space_file = os.path.join(parent_dir, 'gas_phase_space_Rall_Tall_Oref.npy')
    redshift_file = os.path.join(parent_dir, 'gas_history_redshift_Rcore_Thot_Onr.npy')

    phase_space_data = np.load(phase_space_file)
    redshift_history = np.load(redshift_file)

    return phase_space_data, redshift_history


def create_interactive_plot(phase_space_data: np.ndarray,
                            redshift_history: np.ndarray,
                            density_limits: tuple,
                            temperature_limits: tuple
                            ):
    """
    Create an interactive plot of the gas phase space evolution.

    Parameters
    ----------
    phase_space_data : np.ndarray
        3D histogram data with shape (snapshots, density_bins, temperature_bins).
    redshift_history : np.ndarray
        Array of redshift values with length equal to the number of snapshots.
    density_limits : tuple of float
        (min_density, max_density) in units of cm^-3.
    temperature_limits : tuple of float
        (min_temperature, max_temperature) in Kelvin.
    """
    # Compute logarithmic extents for the axes
    log_density_range = np.log10(density_limits)
    log_temperature_range = np.log10(temperature_limits)

    # Set up figure and axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.15)

    # Initial display using the last snapshot
    initial_snapshot = phase_space_data.shape[0] - 1
    image_plot = ax.imshow(
        phase_space_data[initial_snapshot].T,
        norm=LogNorm(),
        interpolation='nearest',
        origin='lower',
        extent=[*log_density_range, *log_temperature_range]
    )

    # Add annotation for redshift
    redshift_annotation = ax.text(
        0.01, 0.99,
        rf'$z = {redshift_history[initial_snapshot]:.2f}$',
        transform=ax.transAxes,
        ha='left', va='top'
    )

    # Label axes
    ax.set_xlabel('Log$_{10}$ Density [cm$^{-3}$]')
    ax.set_ylabel('Log$_{10}$ Temperature [K]')

    # Create slider axis
    slider_ax = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    snapshot_slider = Slider(
        ax=slider_ax,
        label='Snapshot',
        valmin=0,
        valmax=phase_space_data.shape[0] - 1,
        valinit=initial_snapshot,
        valstep=1,
        orientation='vertical'
    )

    def update_plot(val):
        """Update the image and annotation when the slider is moved."""
        idx = int(snapshot_slider.val)
        image_plot.set_data(phase_space_data[idx].T)
        redshift_annotation.set_text(rf'$z = {redshift_history[idx]:.2f}$')
        fig.canvas.draw_idle()

    snapshot_slider.on_changed(update_plot)

    # Create reset button
    reset_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')
    reset_button.on_clicked(lambda event: snapshot_slider.reset())

    plt.show()


def main():
    """
    Main function to load data and launch the interactive plot.
    """
    base_dir = os.path.dirname(__file__)
    density_limits = (1e-8, 1e3)
    temperature_limits = (1e2, 1e10)

    phase_space_data, redshift_history = load_phase_space_data(base_dir)
    create_interactive_plot(
        phase_space_data,
        redshift_history,
        density_limits,
        temperature_limits
    )


if __name__ == '__main__':
    main()
