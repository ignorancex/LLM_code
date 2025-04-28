from dataclasses import asdict

import matplotlib.pyplot as plt

from src.demo.paper.transfer_matrix.utils import visualize_separate
from src.interface.configuration import load_config
from src.outputs.visualization import RcParamsOptions, SubplotsOptions, savefig_dir_list


def main():
    ds_ids = [3, 4]
    vmaxs = [2e10, 1e2]

    config = load_config()
    db = config.database()

    for ds_id, vmax in zip(ds_ids, vmaxs):
        interferograms = db.dataset_interferogram(ds_id=ds_id)

        # VISUALIZE
        rc_params = RcParamsOptions(fontsize=20)
        subplots_opts = SubplotsOptions(figsize=(7.5, 4.8))
        plt.rcParams['font.size'] = str(rc_params.fontsize)
        fig, axs = plt.subplots(**asdict(subplots_opts))
        interferograms.visualize_matrix(
            fig=fig,
            axs=axs[0, 0],
            title="",
            aspect="auto",
            is_colorbar=True,
            y_ticks_decimals=2,
            vmin=0,
            vmax=vmax,
        )

        plt.show()

        # SAVE
        filename = "dataset.pdf"
        project_dir = load_config().directory_paths.project
        paper_dir = project_dir.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"
        figures_dir_list = [
            paper_dir / "figures" / "real",
        ]
        save_subdir = f"invert_mc{interferograms.data.shape[-1]:.0f}/imspoc_uv_2_mc451/dataset"
        savefig_dir_list(
            fig=fig,
            filename=filename,
            directories_list=figures_dir_list,
            subdirectory=save_subdir,
        )


if __name__ == "__main__":
    main()
