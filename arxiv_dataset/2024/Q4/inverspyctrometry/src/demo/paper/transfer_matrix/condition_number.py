from dataclasses import asdict

from matplotlib import pyplot as plt

from src.demo.paper.transfer_matrix.utils import plot_condition_numbers
from src.interface.configuration import load_config
from src.outputs.visualization import savefig_dir_list, SubplotsOptions, RcParamsOptions


def main():
    opd_schema = {"num": 51, "step": 0.175}

    rc_params = RcParamsOptions(fontsize=21)
    subplots_opts = SubplotsOptions(figsize=(8, 5))
    plt.rcParams['font.size'] = str(rc_params.fontsize)
    fig, axs = plt.subplots(**asdict(subplots_opts))
    fig, axs = plot_condition_numbers(
        fig,
        axs[0, 0],
        opd_schema=opd_schema,
        reflectivity_range=(0.4, 0.9, 0.01),
    )
    plt.show()

    # SAVE
    filename = "condition_number_reflectivity.pdf"
    project_dir = load_config().directory_paths.project
    paper_dir = project_dir.parents[1] / "latex" / "20249999_ieee_tsp_inversion_v4"
    figures_dir_list = [
        paper_dir / "figures" / "direct_model",
    ]
    save_subdir = ""
    savefig_dir_list(
        fig=fig,
        filename=filename,
        directories_list=figures_dir_list,
        subdirectory=save_subdir,
    )


if __name__ == "__main__":
    main()
