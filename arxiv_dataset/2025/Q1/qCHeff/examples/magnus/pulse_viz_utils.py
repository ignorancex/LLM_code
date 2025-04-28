import matplotlib.pyplot as plt
import numpy as np

from qcheff.magnus.pulses import ControlPulse, FourierPulse


def plot_basis_funcs(gate_time=2, ax=None):
    # Generate a time list
    coeffs = [1] * 5
    tlist = np.linspace(0, gate_time, 1000)
    pulse = FourierPulse(coeffs, gate_time)

    if ax is None:
        fig, ax = plt.subplots(1, 1, layout="constrained")

    # Plot each basis function
    for i, basis_func in enumerate(pulse.basis_funcs):
        ax.plot(tlist, basis_func(tlist), label=f"n={i+1}")
    ax.set(
        xlabel="Time",
        ylabel="Amplitude",
        title="Modulated Fourier Basis Functions",
        xlim=(0, gate_time),
        ylim=(-1.1, 1.1),
    )
    ax.legend(loc="lower right")
    if ax is None:
        plt.show()


def plot_example_control_pulse(
    gate_time=2, frequency=5.0, phase=0.0, amplitude=1.0, ax=None
):
    # Generate a time list
    coeffs = [1] + [0] * 4
    tlist = np.linspace(0, gate_time, 1000)
    envelope = FourierPulse(coeffs, gate_time)
    pulse = ControlPulse(envelope, frequency, phase, amplitude)

    if ax is None:
        fig, ax = plt.subplots(1, 1, layout="constrained")

    # Plot the control pulse
    ax.plot(tlist, pulse(tlist), label="Control Pulse")
    ax.set(
        xlabel="Time",
        ylabel="Amplitude",
        title="Control Pulse with Fourier Envelope",
        xlim=(0, gate_time),
        ylim=(np.min(pulse(tlist)) - 0.1, np.max(pulse(tlist)) + 0.1),
    )
    ax.legend(loc="lower right")
    if ax is None:
        plt.show()


if __name__ == "__main__":
    plot_basis_funcs()
    plot_example_control_pulse()
