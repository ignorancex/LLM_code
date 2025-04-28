import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.custom_vars import NormOperatorType, LinearOperatorMethod
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.direct_model.interferometer import FabryPerotInterferometer, simulate_interferogram
from src.inverse_model.operators import CTVOperator, NormOperator, LinearOperator
from src.inverse_model.protocols import LorisVerhoeven


def main_other():
    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False)

    wavenumbers = np.linspace(start=0, stop=10*np.pi, num=1000)
    radiance = np.cos(wavenumbers)[:, None]
    spectrum = Spectrum(data=radiance, wavenumbers=wavenumbers)

    spectrum.visualize(axs=axs[0, 0], acq_ind=0)

    opds = 0.175 * np.arange(319)
    transmittance_coefficients = np.array([[1.]])
    reflectance_coefficients = np.array([[0.13]])
    phase_shift = np.zeros_like(opds)
    interferometer = FabryPerotInterferometer(
        transmittance_coefficients=transmittance_coefficients,
        opds=opds,
        reflectance_coefficients=reflectance_coefficients,
        phase_shift=phase_shift,
    )

    transfer_matrix = interferometer.transmittance_response(wavenumbers=wavenumbers)
    interferogram = simulate_interferogram(transmittance_response=transfer_matrix, spectrum=spectrum)

    interferogram.visualize(axs=axs[0, 1], acq_ind=0)

    inverter = LorisVerhoeven(
        regularization_parameter=100.,
        prox_functional=CTVOperator(norm=NormOperator.from_norm(norm=NormOperatorType.L1O)),
        domain_transform=LinearOperator.from_method(method=LinearOperatorMethod.DCT),
        nb_iters=int(500)
    )

    spectrum_rec = inverter.reconstruct_spectrum(interferogram=interferogram, transmittance_response=transfer_matrix)

    spectrum_rec.visualize(axs=axs[0, 2], acq_ind=0)

    plt.show()


def main():
    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False)

    opds = 0.175 * np.arange(319)
    interferogram = Interferogram(data=np.cos(opds)[:, None], opds=opds)
    interferogram.visualize(axs=axs[0, 0], acq_ind=0)

    transmittance_coefficients = np.array([[1.]])
    reflectance_coefficients = np.array([[0.13]])
    phase_shift = np.zeros_like(opds)
    interferometer = FabryPerotInterferometer(
        transmittance_coefficients=transmittance_coefficients,
        opds=opds,
        reflectance_coefficients=reflectance_coefficients,
        phase_shift=phase_shift,
    )

    wavenumbers = np.linspace(start=0, stop=10*np.pi, num=276)
    transfer_matrix = interferometer.transmittance_response(wavenumbers=wavenumbers)

    inverter = LorisVerhoeven(
        regularization_parameter=100.,
        prox_functional=CTVOperator(norm=NormOperator.from_norm(norm=NormOperatorType.L1O)),
        domain_transform=LinearOperator.from_method(method=LinearOperatorMethod.DCT),
        nb_iters=int(500)
    )

    spectrum_rec = inverter.reconstruct_spectrum(interferogram=interferogram, transmittance_response=transfer_matrix)
    spectrum_rec.visualize(axs=axs[0, 1], acq_ind=0)

    plt.show()


if __name__ == "__main__":
    main_other()
