from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.custom_vars import InterferometerType
from src.common_utils.utils import rescale, min_max_normalize, center
from src.demo.paper.transfer_matrix.utils import SamplingOptionsSchema
from src.interface.configuration import load_config
from src.outputs.visualization import savefig_dir_list


def main():
    main_opd_sampling_factor()


def main_opd_sampling_factor():
    opd_sampling_factors = [1, 2]

    for opd_sampling_factor in opd_sampling_factors:
        sampling_options_schema = {
            "experiment_title": "Fabry-Perot",
            "device": {
                "type": InterferometerType.FABRY_PEROT,
                "reflectance_scalar": 0.03,
                "opds": {
                    "num": 50 * opd_sampling_factor + 1,
                    "step": 0.2 / opd_sampling_factor,
                },
            },
            "spectral_range": {
                "min": 0,
                "max": 2.5,
                "override_harmonic_order": None,
            },
        }
        dct_options = {
            "opd_idx": -1,
            "is_rows": True,
        }

        options = SamplingOptionsSchema(**sampling_options_schema)
        experiment = options.create_experiment()

        transfer_matrix = experiment.transfer_matrix()
        transfer_matrix_decomposition = experiment.transfer_matrix_decomposition()

        fig, axs = plt.subplots(nrows=2, ncols=len(transfer_matrix_decomposition) - 1, squeeze=False, tight_layout=True, figsize=(12.8, 9.6))
        transfer_matrix.visualize(
            fig=fig,
            axs=axs[0, 0],
            aspect="auto",
            title=f"{options.experiment_title}   "
                  f"R={options.device.reflectance_scalar:.2f}   "
                  rf"$\Delta\delta={options.device.opds.step:.2f}$",
        )
        transfer_matrix.visualize_dct(
            axs=axs[0, 1],
            title="DCT of A",
            **dct_options,
        )
        for idx in range(1, len(transfer_matrix_decomposition)):
            transfer_matrix_decomposition[idx].visualize_dct(
                axs=axs[1, idx - 1],
                title="DCT of A" + r"$^{(" + f"{idx}" + r")}$",
                ylim=axs[0, 1].get_ylim(),
                **dct_options,
            )

        # savefig_dir_list(
        #     fig=fig,
        #     filename=f"opd_sampling_factor_{int(opd_sampling_factor):02}.png",
        #     directories_list=[load_config().directory_paths.reports],
        #     subdirectory="transfer_matrix/opd_sampling_factors",
        #     fmt="png",
        #     bbox_inches="tight",
        # )

    plt.show()


if __name__ == "__main__":
    main()
