from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.common_utils.custom_vars import Wvn, Acq
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.demo.paper.snr.utils import experiment_dir_convention
from src.interface.configuration import load_config
from src.outputs.serialize import numpy_load_list


def print_metrics(nb_tabs, category, best_idx, lambdaa, rmse_full, rmse_diagonal, rmcw, wavenumbers_size):
    tabs = "\t" * nb_tabs
    print(
        f"{tabs}"
        f"{category}: {best_idx:6},\t"
        f"Lambda: {lambdaa:7.3f},\t"
        f"RMSE: {rmse_full:.3f},\t"
        f"RMSE_DIAG: {rmse_diagonal:.3f},\t"
        f"RMCW: {rmcw:3}/{wavenumbers_size:3} ({rmcw / wavenumbers_size:6.3f})"
    )


def matching_central_wavenumbers_indicator(
        monochromatic_array: np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]],
) -> np.ndarray[bool]:
    acquisition_indices = np.arange(monochromatic_array.shape[-1])
    column_wise_argmax = np.argmax(monochromatic_array, axis=-2)
    is_matching_central_wavenumbers = np.equal(column_wise_argmax, acquisition_indices)
    return is_matching_central_wavenumbers  # Indicator function / vector


def calculate_rmcw(
        monochromatic_array: np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]],
) -> np.ndarray[..., np.dtype[np.int_]]:
    # TODO: Could be useful to have a MonochromaticSpectrum(Spectrum) class here
    """
    Number of Matching Central Wavenumbers:
      The maximum of each acquisition (column) is expected to match with the index of said acquisition.
    """
    acquisition_indices = np.arange(monochromatic_array.shape[-1])
    column_wise_argmax = np.argmax(monochromatic_array, axis=-2)
    rmcw = np.sum(column_wise_argmax == acquisition_indices, axis=-1, dtype=int)
    return rmcw


def split_matching_central_wavenumbers(
        monochromatic_array: np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]],
        target_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    # TODO: Could be useful to have a MonochromaticSpectrum(Spectrum) class here
    if target_type == "diagonal":
        target_array = np.diag(monochromatic_array)
    elif target_type == "maxima":
        target_array = np.max(monochromatic_array, axis=-2)
    else:
        raise ValueError(f"Split type '{target_type}' is not supported.")

    is_matching = matching_central_wavenumbers_indicator(monochromatic_array)
    matching = np.where(is_matching, target_array, np.nan)
    mismatching = np.where(~is_matching, target_array, np.nan)

    return matching, mismatching


def visualize_matching_central_wavenumbers(
        spectra: Spectrum,
        axs,
        target_type: str,
        linestyle: str = "-",
        title: str = None,
        xlabel: str = None,
        xlim: list = None,
        ylabel: str = None,
        ylim: list = None,
):
    # TODO: Could be useful to have a MonochromaticSpectrum(Spectrum) class here
    matching, mismatching = split_matching_central_wavenumbers(spectra.data, target_type=target_type)

    axs.plot(
        spectra.wavenumbers,
        np.ones(shape=spectra.data.shape[-1]),
        linestyle=linestyle,
        label="Reference",
        linewidth=3,
    )
    axs.plot(
        spectra.wavenumbers,
        matching,
        linestyle=linestyle,
        label="Matching",
    )
    axs.plot(
        spectra.wavenumbers,
        mismatching,
        linestyle=linestyle,
        label="Mismatching",
    )

    if title is None:
        title = "Matching Central Wavenumbers"
    if xlabel is None:
        xlabel = rf"Associated Central Wavenumbers [{spectra.wavenumbers_unit}]"
    if ylabel is None:
        ylabel = "Intensity"
    if xlim is not None:
        axs.set_ylim(xlim)
    if ylim is not None:
        axs.set_ylim(ylim)

    axs.set_title(title)
    axs.set_ylabel(ylabel)
    axs.set_xlabel(xlabel)
    axs.legend()
    axs.grid(visible=True)


def crop_spectrum_wavenumbers(
        spectrum: Spectrum,
        new_range: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
) -> Spectrum:
    mask = (new_range[0] <= spectrum.wavenumbers) & (spectrum.wavenumbers <= new_range[-1])
    new_wavenumbers = spectrum.wavenumbers[mask]
    new_data = spectrum.data[mask][:, mask]
    return replace(spectrum, data=new_data, wavenumbers=new_wavenumbers)


def crop_interferogram_acquisitions(
        interferogram: Interferogram,
        central_wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
        new_range: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
) -> Interferogram:
    mask = (new_range[0] <= central_wavenumbers) & (central_wavenumbers <= new_range[-1])
    new_data = interferogram.data[:, mask]
    return replace(interferogram, data=new_data)


def experiment_subdir_convention(
        dataset_id: int,
        characterization_id: int = -1,
        noise_level_index: int = -1,
        inversion_protocol_id: int = -1,
):
    config = load_config()
    db = config.database()
    subdir = f"invert_{db.datasets[dataset_id].title}"
    if characterization_id >= 0:
        subdir = f"{subdir}/{db.characterizations[characterization_id].title}"
    if noise_level_index >= 0:
        subdir = f"{subdir}/{int(db.noise_levels[noise_level_index])}_db"
    if inversion_protocol_id >= 0:
        subdir = f"{subdir}/{db.inversion_protocols[inversion_protocol_id].title}"
    return subdir


def metrics_full_table(experiment_id: int):
    config = load_config()
    db = config.database()
    experiment_config = db.experiments[experiment_id]

    nb_dss = len(experiment_config.dataset_ids)
    nb_ifms = len(experiment_config.interferometer_ids)
    nb_nls = len(experiment_config.noise_level_indices)
    nb_ips = len(experiment_config.inversion_protocol_ids)
    full_table = np.zeros(shape=(nb_dss * nb_ips, nb_ifms * nb_nls * 4))

    index = []
    for _ in experiment_config.dataset_ids:
        for ip_id in experiment_config.inversion_protocol_ids:
            gls_str = "& \\gls{" + f"{db.inversion_protocols[ip_id].title.lower()}" + "}"
            index.append(gls_str)

    header = []
    for _ in experiment_config.interferometer_ids:
        for _ in experiment_config.noise_level_indices:
            header.append("lambda")
            header.append("rmse diag")
            header.append("rmse full")
            header.append("rmcw")

    for i_ds, ds_id in enumerate(experiment_config.dataset_ids):
        for i_ifm, ifm_id in enumerate(experiment_config.interferometer_ids):
            for i_nl, nl_idx in enumerate(experiment_config.noise_level_indices):
                for i_ip, ip_id in enumerate(experiment_config.inversion_protocol_ids):
                    lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=ip_id)
                    metrics_dir = experiment_dir_convention(dir_type="metrics", experiment_id=experiment_id)
                    inverter_subdir = experiment_subdir_convention(
                        dataset_id=ds_id,
                        characterization_id=ifm_id,
                        noise_level_index=nl_idx,
                        inversion_protocol_id=ip_id,
                    )

                    rmse_diagonal, rmse_full, rmcw = numpy_load_list(
                        filenames=["rmse_diagonal.npy", "rmse_full.npy", "rmcw.npy"],
                        directory=metrics_dir,
                        subdirectory=inverter_subdir,
                    )
                    argmin = np.argmin(rmse_diagonal)
                    lambdaa_min = lambdaas[argmin]
                    rmse_diagonal_min = rmse_diagonal[argmin]
                    rmse_full_min = rmse_full[argmin]
                    rmcw_min = rmcw[argmin]
                    if lambdaa_min == 0:
                        lambdaa_min = np.nan

                    full_table[i_ip + nb_ips*i_ds, 2*i_nl + 2*nb_nls*i_ifm] = lambdaa_min
                    full_table[i_ip + nb_ips*i_ds, 2*i_nl + 2*nb_nls*i_ifm + 1] = rmse_diagonal_min
                    full_table[i_ip + nb_ips*i_ds, 2*i_nl + 2*nb_nls*i_ifm + 2] = rmse_full_min
                    full_table[i_ip + nb_ips*i_ds, 2*i_nl + 2*nb_nls*i_ifm + 3] = rmcw_min

    return full_table, header, index
