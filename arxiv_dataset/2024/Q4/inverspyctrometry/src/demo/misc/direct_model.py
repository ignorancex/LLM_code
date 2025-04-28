import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.custom_vars import InterferometerType as IfmType
from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import generate_sampled_opds, generate_wavenumbers_from_opds
from src.direct_model.interferometer import interferometer_factory


def main():
    """
    For a future slider, my variables are:
    - Reflectance: To increase N as the N-wave approximation in FP with order 0
    - OPD index in the plot of the transfer matrices: To observe the OPD responses and their DCT
    - OPD value in the cosines that constitute the spectral radiance: Taken from the list of OPDs for controlled observation of the interferograms
    - Acquisition index for plot purposes
    """
    acq_ind = 0
    nb_opd, del_opd = 320, 0.175
    opds = generate_sampled_opds(nb_opd=nb_opd, opd_step=del_opd)
    nb_wn = opds.size*6  # quasi-continuous
    wavenumbers = generate_wavenumbers_from_opds(wavenumbers_num=nb_wn, del_opd=del_opd)

    reflectance_coefficients = 0.15 * np.ones(shape=(opds.size, 1))
    transmittance_coefficients = 1 - reflectance_coefficients
    phase_shift = np.zeros_like(a=opds)

    plot_opd_idx = 4
    radiance_cosine_args = {
        "amplitudes": np.array([3, 2, 1]),
        "opds": opds[[plot_opd_idx, 150, 300]],
    }

    michelson = interferometer_factory(
        option=IfmType.MICHELSON,
        transmittance_coefficients=transmittance_coefficients,
        opds=opds,
        phase_shift=phase_shift,
        reflectance_coefficients=reflectance_coefficients,
        order=0,
    )
    fabry_perot = interferometer_factory(
        option=IfmType.FABRY_PEROT,
        transmittance_coefficients=transmittance_coefficients,
        opds=opds,
        phase_shift=phase_shift,
        reflectance_coefficients=reflectance_coefficients,
        order=0,
    )
    fabry_perot_2 = interferometer_factory(
        option=IfmType.FABRY_PEROT,
        transmittance_coefficients=transmittance_coefficients,
        opds=opds,
        phase_shift=phase_shift,
        reflectance_coefficients=reflectance_coefficients,
        order=2,
    )

    transfer_matrix_mich = michelson.transmittance_response(wavenumbers=wavenumbers)
    transfer_matrix_fp = fabry_perot.transmittance_response(wavenumbers=wavenumbers)
    transfer_matrix_fp_2 = fabry_perot_2.transmittance_response(wavenumbers=wavenumbers)

    spectrum = Spectrum.from_oscillations(
        amplitudes=radiance_cosine_args["amplitudes"],
        opds=radiance_cosine_args["opds"],
        wavenumbers=wavenumbers,
    )

    interferogram_mich = michelson.acquire_interferogram(spectrum=spectrum)
    interferogram_fp = fabry_perot.acquire_interferogram(spectrum=spectrum)
    interferogram_fp_2 = fabry_perot_2.acquire_interferogram(spectrum=spectrum)

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False)
    transfer_matrix_mich.visualize(fig=fig, axs=axs[0, 0])
    transfer_matrix_mich.visualize_opd_response(axs=axs[0, 1], opd_idx=plot_opd_idx)
    transfer_matrix_mich.visualize_dct(axs=axs[1, 0], opd_idx=-1)
    transfer_matrix_mich.visualize_singular_values(axs=axs[1, 1])

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False)
    transfer_matrix_fp_2.visualize(fig=fig, axs=axs[0, 0])
    transfer_matrix_fp_2.visualize_opd_response(axs=axs[0, 1], opd_idx=plot_opd_idx)
    transfer_matrix_fp_2.visualize_dct(axs=axs[1, 0], opd_idx=-1)
    transfer_matrix_fp_2.visualize_singular_values(axs=axs[1, 1])

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False)
    transfer_matrix_fp.visualize(fig=fig, axs=axs[0, 0])
    transfer_matrix_fp.visualize_opd_response(axs=axs[0, 1], opd_idx=plot_opd_idx)
    transfer_matrix_fp.visualize_dct(axs=axs[1, 0], opd_idx=-1)
    transfer_matrix_fp.visualize_singular_values(axs=axs[1, 1])

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False)
    spectrum.visualize(axs=axs[0, 0], acq_ind=acq_ind)
    interferogram_mich.visualize(axs=axs[0, 1], acq_ind=acq_ind)
    interferogram_fp.visualize(axs=axs[1, 0], acq_ind=acq_ind)
    interferogram_fp_2.visualize(axs=axs[1, 1], acq_ind=acq_ind)

    plt.show()


if __name__ == "__main__":
    main()
