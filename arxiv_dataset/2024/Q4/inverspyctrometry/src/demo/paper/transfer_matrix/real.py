from dataclasses import asdict

import matplotlib.pyplot as plt

from src.demo.paper.transfer_matrix.utils import visualize_separate
from src.interface.configuration import load_config
from src.outputs.visualization import RcParamsOptions, SubplotsOptions, savefig_dir_list


def main():
    char_id = 0

    config = load_config()
    db = config.database()
    characterization = db.characterization(characterization_id=char_id)

    wavenumbers = db.characterization_wavenumbers(char_id=char_id)
    transmittance = characterization.transmittance(wavenumbers=wavenumbers)
    reflectivity = characterization.reflectance(wavenumbers=wavenumbers)
    airy_gain = transmittance * (1 - reflectivity) * (1 + reflectivity)

    transfer_matrix = characterization.transmittance_response(wavenumbers=wavenumbers)
    dct_orthogonalize_kwargs = {
        "device_type": characterization.interferometer_type,
        "reflectance": reflectivity,
        "airy_gain": airy_gain,
    }

    # VISUALIZE
    rc_params = RcParamsOptions(fontsize=20)
    subplots_opts = SubplotsOptions(figsize=(7.5, 4.8))
    plt.rcParams['font.size'] = str(rc_params.fontsize)
    figs, axes = zip(*[plt.subplots(**asdict(subplots_opts)) for _ in range(4)])
    figs, axes = visualize_separate(
        figs=figs,
        axes=axes,
        transfer_matrix=transfer_matrix,
        dct_orthogonalize_kwargs=dct_orthogonalize_kwargs,
        opd_idx=20,
        is_show=False,
        x_ticks_decimals=2,
        y_ticks_decimals=2,
        markevery=30,
        vmin=0,
        vmax=2e10,
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
    save_subdir = f"imspoc_uv_2_mc451/transfer_matrices"
    for filename, fig in zip(filenames, figs):
        savefig_dir_list(
            fig=fig,
            filename=filename,
            directories_list=figures_dir_list,
            subdirectory=save_subdir,
        )


if __name__ == "__main__":
    main()
