from pathlib import Path
from typing import Optional

import numpy as np

from src.interface.configuration import load_config
from src.outputs.serialize import numpy_load_list


def experiment_dir_convention(
        dir_type: str,
        experiment_id: int,
        custom_dir: Optional[Path] = None,
) -> Path:
    """If no custom_dir is provided, the default is the reports folder"""
    config = load_config()
    if custom_dir is None:
        custom_dir = config.directory_paths.reports

    if dir_type == "simulation":
        return custom_dir / f"experiment_{experiment_id}" / "simulation"
    elif dir_type == "reconstruction":
        return custom_dir / f"experiment_{experiment_id}" / "reconstruction"
    elif dir_type == "metrics":
        return custom_dir / f"experiment_{experiment_id}" / "metrics"
    elif dir_type == "figures":
        return custom_dir / f"experiment_{experiment_id}" / "figures"
    elif dir_type == "paper_figures":
        db = config.database()
        return custom_dir / "figures" / f"{db.experiments[experiment_id].type}"
    else:
        raise ValueError()


def experiment_subdir_convention(
        dataset_id: int,
        interferometer_id: int = -1,
        noise_level_index: int = -1,
        inversion_protocol_id: int = -1,
):
    config = load_config()
    db = config.database()
    subdir = f"invert_{db.datasets[dataset_id].title}"
    if interferometer_id >= 0:
        subdir = f"{subdir}/{db.interferometers[interferometer_id].title}"
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
    full_table = np.zeros(shape=(nb_dss * nb_ips, nb_ifms * nb_nls * 2))

    index = []
    for _ in experiment_config.dataset_ids:
        for ip_id in experiment_config.inversion_protocol_ids:
            gls_str = "& \\gls{" + f"{db.inversion_protocols[ip_id].title.lower()}" + "}"
            index.append(gls_str)

    header = []
    for _ in experiment_config.interferometer_ids:
        for _ in experiment_config.noise_level_indices:
            header.append("lambda")
            header.append("rmse")

    for i_ds, ds_id in enumerate(experiment_config.dataset_ids):
        for i_ifm, ifm_id in enumerate(experiment_config.interferometer_ids):
            for i_nl, nl_idx in enumerate(experiment_config.noise_level_indices):
                for i_ip, ip_id in enumerate(experiment_config.inversion_protocol_ids):
                    metrics_dir = experiment_dir_convention(dir_type="metrics", experiment_id=experiment_id)
                    inverter_subdir = experiment_subdir_convention(
                        dataset_id=ds_id,
                        interferometer_id=ifm_id,
                        noise_level_index=nl_idx,
                        inversion_protocol_id=ip_id,
                    )

                    lambdaa_min, rmse_min = numpy_load_list(
                        filenames=["lambdaa_min.npy", "rmse_min.npy"],
                        directory=metrics_dir,
                        subdirectory=inverter_subdir,
                    )
                    if lambdaa_min == 0:
                        lambdaa_min = np.nan

                    full_table[i_ip + nb_ips*i_ds, 2*i_nl + 2*nb_nls*i_ifm] = lambdaa_min
                    full_table[i_ip + nb_ips*i_ds, 2*i_nl + 2*nb_nls*i_ifm + 1] = rmse_min

    return full_table, header, index
