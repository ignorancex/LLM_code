from dataclasses import replace, dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from src.common_utils.function_generator import GaussianGenerator
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import calculate_rmse, polyval_rows, match_stats
from src.direct_model.interferometer import FabryPerotInterferometer
from src.interface.configuration import load_config
from src.inverse_model.analytical_inverter import calculate_airy_fourier_coeffs, HaarInverter


def compute_wavenumbers(opd, wn, wn_num_factor):
    opd_max = opd.num * opd.step
    wn_step_min = 1 / (2 * opd_max)  # Nyquist-Shannon bandwidth
    wn_step = wn_step_min / wn_num_factor  # oversampling to mimic continuity
    wavenumbers = np.arange(wn.start, wn.stop, wn_step)
    return wavenumbers


def oversample_wavenumbers(wavenumbers, factor):
    original_points = len(wavenumbers)
    new_points = original_points * factor
    original_indices = np.arange(original_points)
    new_indices = np.linspace(0, original_points - 1, new_points)
    interp_func = interp1d(original_indices, wavenumbers, kind='linear')
    oversampled_wavenumbers = interp_func(new_indices)
    return oversampled_wavenumbers


def generate_synthetic_spectrum(gauss, opd, wn):
    opd_max = opd.num * opd.step
    wn_step = 1 / (2 * opd_max)  # Nyquist-Shannon bandwidth

    wavenumbers = np.arange(wn.start, wn.stop, wn_step)
    gaussian_gen = GaussianGenerator(
        coefficients=gauss.coeffs[:, None],
        means=gauss.means[:, None],
        stds=gauss.stds[:, None],
    )
    spectrum_data = gaussian_gen.generate_data(variable=wavenumbers)
    spectrum_ref = Spectrum(data=spectrum_data, wavenumbers=wavenumbers, wavenumbers_unit=wn.unit)

    return spectrum_ref


@dataclass
class OPDummy:
    data: np.ndarray
    unit: str

    @property
    def step(self):
        return np.mean(np.diff(self.data))

    @property
    def num(self):
        return self.data.size

    @property
    def start(self):
        return self.data[0]

    @property
    def max(self):
        return self.data[-1]

    @classmethod
    def from_opd_info(cls, step, num, unit):
        opds_arr = np.arange(0, step * num, step)
        return cls(data=opds_arr, unit=unit)


def oversample_spectrum(spectrum_reference, opds: OPDummy, wn_continuity_factor):
    wn_step = np.mean(np.diff(spectrum_reference.wavenumbers))
    wn_step_nyquist = 1 / (2 * opds.max)
    wn_step_target = wn_step_nyquist / wn_continuity_factor
    wn_factor_new = int(np.ceil(wn_step / wn_step_target))
    wavenumbers_new = oversample_wavenumbers(spectrum_reference.wavenumbers, factor=wn_factor_new)
    spectrum_continuous = spectrum_reference.interpolate(wavenumbers=wavenumbers_new, kind="slinear")
    return spectrum_continuous


def generate_interferogram(opds: OPDummy, fp, spectrum_reference, wn_continuity_factor):
    device = FabryPerotInterferometer(
        transmittance_coefficients=fp.transmittance,
        opds=opds.data,
        phase_shift=fp.phase_shift,
        reflectance_coefficients=fp.reflectance,
        order=fp.order,
    )

    spectrum_continuous = oversample_spectrum(spectrum_reference, opds, wn_continuity_factor)
    interferogram = device.acquire_interferogram(spectrum_continuous)

    wn_step = np.mean(np.diff(spectrum_continuous.wavenumbers))
    interferogram = replace(interferogram, data=interferogram.data * wn_step, opds_unit=opds.unit)

    return interferogram


def assert_haar_check(fp, interferogram_dft, spectrum, haar_order):
    k_vals = np.where(
        np.logical_and(
            spectrum.wavenumbers[0] <= interferogram_dft.wavenumbers,
            interferogram_dft.wavenumbers <= spectrum.wavenumbers[-1],
        )
    )

    a_cap = calculate_airy_fourier_coeffs(fp, haar_order)

    k_cap = interferogram_dft.wavenumbers.size
    b_matrix = np.zeros(shape=(k_cap, k_cap))
    for mu in k_vals[0]:
        for n in range(1, haar_order + 1):
            for k in k_vals[0]:
                for i in range(n):
                    if n*k+i == mu:
                        b_matrix[mu, k] += a_cap[n] / n

    spectrum_interp = spectrum.interpolate(wavenumbers=interferogram_dft.wavenumbers, kind="cubic")

    spectrum_transform = replace(spectrum_interp, data=b_matrix @ spectrum_interp.data)

    return spectrum_transform, b_matrix


def invert_haar(wavenumbers, fp, haar_order, interferogram: Interferogram):
    transmissivity = polyval_rows(coefficients=fp.transmittance, interval=wavenumbers).mean(axis=-1)
    reflectivity = polyval_rows(coefficients=fp.reflectance, interval=wavenumbers).mean(axis=-1)

    haar = HaarInverter(
        transmittance=transmissivity,
        wavenumbers=wavenumbers,
        reflectance=reflectivity,
        order=haar_order,
        is_mean_center=True,
    )
    spectrum = haar.reconstruct_spectrum(interferogram=interferogram)
    return spectrum


def load_spectrum(option: str):
    config = load_config()
    db = config.database()
    if option == "solar":
        spectrum = db.dataset_spectrum(ds_id=0)
        # acq_id = 0
    elif option in ["specim", "cc_green"]:
        spectrum = db.dataset_spectrum(ds_id=1)
        # acq_id = 13
    elif option == "mc451":
        central_wavenumbers = db.dataset_central_wavenumbers(dataset_id=3)
        spectrum = Spectrum(data=np.eye(central_wavenumbers.size), wavenumbers=central_wavenumbers)
        # acq_id = 200
    elif option == "mc651":
        central_wavenumbers = db.dataset_central_wavenumbers(dataset_id=4)
        spectrum = Spectrum(data=np.eye(central_wavenumbers.size), wavenumbers=central_wavenumbers)
        # acq_id = 300
    else:
        raise ValueError(f"Option {option} is not supported.")
    # spectrum = replace(spectrum, data=spectrum.data[:, acq_id:acq_id + 1])
    return spectrum


def invert_protocols(protocols: list, wavenumbers, fp, interferogram: Interferogram, spectrum_ref: Spectrum):
    device = FabryPerotInterferometer(
        transmittance_coefficients=fp.transmittance,
        opds=interferogram.opds,
        phase_shift=fp.phase_shift,
        reflectance_coefficients=fp.reflectance,
        order=fp.order,
    )
    transmittance_response = device.transmittance_response(wavenumbers=wavenumbers)
    transmittance_response = transmittance_response.rescale(new_max=1., axis=None)
    fig, axs = plt.subplots()
    transmittance_response.visualize(fig, axs)
    plt.show()

    db = load_config().database()
    spectrum_protocols = []
    argmin_rmses = []
    for protocol in protocols:
        if protocol.label != "HAAR":
            lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=protocol.id)
            spectrum_rec_all = np.zeros(shape=(lambdaas.size, *spectrum_ref.data.shape))
            for i_lmd, lambdaa in enumerate(lambdaas):
                inverter = db.inversion_protocol(inv_protocol_id=protocol.id, lambdaa=lambdaa)
                spectra_rec = inverter.reconstruct_spectrum(
                    interferogram=interferogram, transmittance_response=transmittance_response
                )
                spectrum_rec_all[i_lmd] = spectra_rec.data

            rmse_lambdaas = calculate_rmse(
                array=spectrum_rec_all,
                reference=spectrum_ref.data,
                is_match_axis=-2,
                is_match_stats=True,
                is_rescale_reference=True,
            )
            argmin_rmse = np.argmin(rmse_lambdaas)
            print(f"{protocol.label}: lmd = {lambdaas[argmin_rmse]:.4f} at idx = {argmin_rmse:.0f}")
            spectrum_rec_best = replace(spectrum_ref, data=spectrum_rec_all[argmin_rmse])

            spectrum_protocols.append(spectrum_rec_best)
            argmin_rmses.append(argmin_rmse)

    return spectrum_protocols, argmin_rmses


@dataclass(frozen=True)
class Protocol:
    id: int
    label: str
    color: str
