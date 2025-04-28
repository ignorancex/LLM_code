from dataclasses import asdict

import matplotlib.pyplot as plt

from src.common_utils.custom_vars import InterferometerType
from src.demo.paper.transfer_matrix.utils import visualize_separate, SamplingOptionsSchema
from src.interface.configuration import load_config
from src.outputs.visualization import RcParamsOptions, SubplotsOptions, savefig_dir_list


def load_paper_options_list():
    """Loads a list of schemas"""

    options_schema_list = [
        {
            "experiment_title": "mich_oversampled",
            "device": {
                "type": InterferometerType.MICHELSON,
                "reflectance_scalar": 0.5,
                "opds": {
                    "num": 51,
                    "step": 0.2,
                },
            },
            "spectral_range": {
                "min": 1.,
                "max": 2.5,
                "override_harmonic_order": 3,
            },
        },
        {
            "experiment_title": "fp_0_low_r",
            "device": {
                "type": InterferometerType.FABRY_PEROT,
                "reflectance_scalar": 0.2,
                "opds": {
                    "num": 51,
                    "step": 0.2,
                },
            },
            "spectral_range": {
                "min": 1.,
                "max": 2.5,
                "override_harmonic_order": None,
            },
        },
        {
            "experiment_title": "fp_0_med_r",
            "device": {
                "type": InterferometerType.FABRY_PEROT,
                "reflectance_scalar": 0.5,
                "opds": {
                    "num": 51,
                    "step": 0.2,
                },
            },
            "spectral_range": {
                "min": 1.,
                "max": 2.5,
                "override_harmonic_order": None,
            },
        },
        {
            "experiment_title": "fp_0_high_r",
            "device": {
                "type": InterferometerType.FABRY_PEROT,
                "reflectance_scalar": 0.8,
                "opds": {
                    "num": 51,
                    "step": 0.2,
                },
            },
            "spectral_range": {
                "min": 1.,
                "max": 2.5,
                "override_harmonic_order": None,
            },
        },
    ]
    return options_schema_list


def main():
    sampling_options_schema_list = load_paper_options_list()

    for sampling_options_schema in sampling_options_schema_list:
        options = SamplingOptionsSchema(**sampling_options_schema)
        experiment = options.create_experiment()
        transmittance = experiment.transmittance()
        reflectivity = experiment.reflectance()
        airy_gain = transmittance ** 2

        transfer_matrix = experiment.transfer_matrix()
        dct_orthogonalize_kwargs = {
            "device_type": experiment.device_type,
            "reflectance": reflectivity,
            "airy_gain": airy_gain,
        }
        alpha = experiment.alpha()

        rc_params = RcParamsOptions(fontsize=21)
        subplots_opts = SubplotsOptions(figsize=(6.4, 4.8))
        plt.rcParams['font.size'] = str(rc_params.fontsize)
        figs, axes = zip(*[plt.subplots(**asdict(subplots_opts)) for _ in range(4)])
        figs, axes = visualize_separate(
            figs=figs,
            axes=axes,
            transfer_matrix=transfer_matrix,
            dct_orthogonalize_kwargs=dct_orthogonalize_kwargs,
            opd_idx=10,
            is_show=True,
            x_ticks_decimals=1,
            y_ticks_decimals=0,
            markevery=5,
            alpha=alpha,
        )

        # SAVE
        filenames = [
            "transfer_matrix.pdf",
            "singular_values.pdf",
            "opd_response.pdf",
            "opd_dct.pdf",
        ]
        project_dir = load_config().directory_paths.project
        paper_dir = project_dir.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"
        figures_dir_list = [
            paper_dir / "figures" / "direct_model",
        ]
        save_subdir = f"{experiment.experiment_title}/transfer_matrices"
        for filename, fig in zip(filenames, figs):
            savefig_dir_list(
                fig=fig,
                filename=filename,
                directories_list=figures_dir_list,
                subdirectory=save_subdir,
            )


if __name__ == "__main__":
    main()
