import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.function_generator import GaussianGenerator
from src.common_utils.light_wave import Spectrum
from src.direct_model.interferometer import FabryPerotInterferometer, MichelsonInterferometer
from src.inverse_model.protocols import IDCT, PseudoInverse


def main():
    paper_test(
        gauss_coeffs=np.array([1., 0.9, 0.75]),
        gauss_means=np.array([2000, 4250, 6500]),  # cm
        gauss_stds=np.array([300, 1125, 400]),  # cm
        opd_step=100*1e-7,  # cm
        opd_num=2048,
        reflectance=np.array([0.7]),
        wn_min=0.,  # cm
        wn_max=20000.1,  # cm
        haar_order=5,
        fp_order=0,
        opd_samples_skip=0,
    )


def paper_test(
        gauss_coeffs,
        gauss_means,
        gauss_stds,
        opd_step: float,
        opd_num: int,
        reflectance,
        wn_min: float,
        wn_max: float,
        haar_order: int,
        opd_samples_skip: int,
        fp_order: int = 0,
):
    """
    Generate a list of OPDs with the lowest being missing.
    Simulate a spectrum.
    Simulate an interferogram.
    Reconstruct the spectrum and compare.
    """
    opds = opd_step * np.arange(opd_num)
    opds[:opd_samples_skip] = 0

    # wn_step = 1 / (2 * opds.max())
    wn_max_dct = 1 / (2 * opd_step)
    wn_num = int(opd_num * (wn_max - wn_min) / wn_max_dct)
    wavenumbers = np.linspace(start=wn_min, stop=wn_max, num=wn_num*10, endpoint=False)  # um-1

    gaussian_gen = GaussianGenerator(
        coefficients=gauss_coeffs[:, None],
        means=gauss_means[:, None],
        stds=gauss_stds[:, None],
    )
    spectrum_data = gaussian_gen.generate_data(variable=wavenumbers)
    spectrum_ref = Spectrum(data=spectrum_data, wavenumbers=wavenumbers)

    transmittance = np.array([1.])  # The values in the paper seem to be normalized by the transmittance

    ifm = FabryPerotInterferometer(
        transmittance_coefficients=transmittance,
        opds=opds,
        phase_shift=np.array([0]),
        reflectance_coefficients=reflectance,
        order=fp_order,
    )
    interferogram = ifm.acquire_interferogram(spectrum=spectrum_ref)

    wn_step = 1 / (2 * opds.max())
    wn_max_dct = 1 / (2 * opd_step)
    wn_num = int(opd_num * (wn_max - wn_min) / wn_max_dct)
    wavenumbers = wn_min + wn_step * np.arange(wn_num)  # um-1

    idct_inv = IDCT(is_mean_center=True)
    transmittance_response = ifm.transmittance_response(wavenumbers=wavenumbers, is_correct_transmittance=False)
    spectrum_idct = idct_inv.reconstruct_spectrum(
        interferogram=interferogram,
        transmittance_response=transmittance_response
    )

    pinv_inv = PseudoInverse()
    transmittance_response = ifm.transmittance_response(wavenumbers=wavenumbers, is_correct_transmittance=False)
    spectrum_pinv = pinv_inv.reconstruct_spectrum(
        interferogram=interferogram,
        transmittance_response=transmittance_response
    )

    acq_ind = 0
    ylim = [-0.2, 1.3]

    fig, axs = plt.subplots(1, 3, squeeze=False)

    spectrum_ref.visualize(axs=axs[0, 0], acq_ind=acq_ind, color="red", label='Reference', ylim=ylim)
    axs[0, 0].set_title('Reference Spectrum')
    axs[0, 0].set_xlabel('Wavenumbers [cm-1]')
    axs[0, 0].set_ylabel('Intensity')

    interferogram.visualize(axs=axs[0, 1], acq_ind=acq_ind)
    axs[0, 1].set_title('Simulated Interferogram')
    axs[0, 1].set_xlabel('OPDs [cm]')
    axs[0, 1].set_ylabel('Intensity')

    spectrum_ref.visualize(axs=axs[0, 2], acq_ind=acq_ind, color="red", label='Reference', ylim=ylim)
    spectrum_idct_eq, _ = spectrum_idct.match_stats(reference=spectrum_ref, axis=-2)
    spectrum_idct.visualize(axs=axs[0, 2], acq_ind=acq_ind, linestyle="dashed", color="green", label='IDCT', ylim=ylim)
    # spectrum_pinv_eq, _ = spectrum_pinv.match_stats(reference=spectrum_ref, axis=-2)
    # spectrum_pinv_eq.visualize(axs=axs[0, 2], acq_ind=acq_ind, linestyle="dashed", color="blue", label='PINV', ylim=ylim)
    axs[0, 2].set_title('Reconstructed Spectrum')
    axs[0, 2].set_xlabel('Wavenumbers [cm-1]')
    axs[0, 2].set_ylabel('Intensity')

    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
