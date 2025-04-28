from src.demo.paper.idct.invert import invert_list_experiments
from src.demo.paper.idct.metrics import metrics_list_experiments
from src.demo.paper.idct.visualize import visualize_list_experiments
from src.outputs.visualization import RcParamsOptions, SubplotsOptions


def run_full_pipeline(
        experiment_ids: list[int],
        is_compensate: bool,
        is_verbose: bool,
        rc_params: RcParamsOptions,
        subplots_options: SubplotsOptions,
        plot_options: dict,
        acquisition_indices: list[int],
        is_plot_show: bool = False,
):
    invert_list_experiments(experiment_ids=experiment_ids, is_compensate=is_compensate)
    metrics_list_experiments(experiment_ids=experiment_ids, is_verbose=is_verbose)
    visualize_list_experiments(
        experiment_ids=experiment_ids,
        rc_params=rc_params,
        subplots_options=subplots_options,
        plot_options=plot_options,
        acquisition_indices=acquisition_indices,
        is_plot_show=is_plot_show,
    )


def main():
    experiment_ids = [3, 4, 5, 6]

    is_compensate = True

    is_verbose = True

    rc_params = RcParamsOptions(fontsize=17)
    subplots_options = SubplotsOptions()
    plot_options = {"ylim": [-0.2, 1.4]}
    acquisition_indices = [0, 13, 13]
    is_plot_show = False

    run_full_pipeline(
        experiment_ids=experiment_ids,
        is_compensate=is_compensate,
        is_verbose=is_verbose,
        rc_params=rc_params,
        subplots_options=subplots_options,
        plot_options=plot_options,
        acquisition_indices=acquisition_indices,
        is_plot_show=is_plot_show,
    )


if __name__ == "__main__":
    main()
