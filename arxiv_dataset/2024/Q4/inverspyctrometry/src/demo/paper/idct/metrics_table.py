from src.common_utils.utils import numpy_to_latex
from src.demo.paper.snr.utils import metrics_full_table, experiment_dir_convention
from src.outputs.serialize import numpy_save_list


def main():
    experiment_id = 5

    metrics_table_array, header, row_labels = metrics_full_table(experiment_id=experiment_id)
    experiment_dir = experiment_dir_convention(dir_type="metrics", experiment_id=experiment_id)
    numpy_save_list(
        filenames=["metrics_table_array.npy"],
        arrays=[metrics_table_array],
        directories=[experiment_dir],
        subdirectory="",
    )

    metrics_table_latex = numpy_to_latex(
        array=metrics_table_array,
        row_labels=row_labels,
        header=header,
        index=True,
        na_rep="-",
        float_format="%.3f",
    )
    print(metrics_table_latex)


if __name__ == "__main__":
    main()
