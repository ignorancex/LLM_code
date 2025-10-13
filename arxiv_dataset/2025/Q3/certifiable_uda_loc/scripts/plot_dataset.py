# Next TODO: Lost in the woods.
# TODO
# Maybe another comparison of Max-Mix GT INIT with GT fixed data associations?

from certifiable_uda_loc.lost_in_the_woods.analysis_liw import (
    get_analyses,
)
import certifiable_uda_loc.analysis_machinery as analysis_machinery
import pandas as pd
import os
from pathlib import Path
from loguru import logger
from matplotlib import pyplot as plt
import seaborn as sns
from certifiable_uda_loc.lost_in_the_woods.analysis_liw import (
    AnalysisLIW,
    default_arg_dict_liw,
)

import certifiable_uda_loc.lost_in_the_woods.dataset as dataset
import certifiable_uda_loc.path_configuration as path_config


def main(analysis: AnalysisLIW):
    top_level_result_dir: str = analysis.top_level_result_dir
    liw_ds_path = path_config.lost_in_the_woods_dataset_path
    liw_ds: dataset.LostInTheWoodsDataset = dataset.LostInTheWoodsDataset.from_mat_file(
        liw_ds_path,
        time_bounds=[0, 60 * 10],
    )
    fig, ax = liw_ds.plot()
    plt.savefig("./liw_ds_plot.pdf")
    # df = analysis.run_analysis(
    #     delete_existing_dataframe=True, postprocess_only=True, continue_trials=False
    # )
    # df = analysis.run_analysis(
    # delete_existing_dataframe=False, postprocess_only=True, continue_trials=False
    # )
    # analysis.plots()
    # print(df.describe())
    # analysis.checks()
    # print(list(df))
    # print(df1.describe())


if __name__ == "__main__":
    for analysis_id, analysis in get_analyses().items():
        # if analysis_id != "high_vs_low_noise":
        # continue
        if analysis_id != "lost_in_the_woods_debug":
            continue
        main(analysis)
