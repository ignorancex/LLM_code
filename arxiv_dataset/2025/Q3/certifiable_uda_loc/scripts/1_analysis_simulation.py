from certifiable_uda_loc.analysis_machinery import (
    get_analyses,
    Analysis,
    default_arg_dict,
    string_id_from_parameters,
)
import certifiable_uda_loc.analysis_machinery as analysis_machinery
import pandas as pd
import os
from pathlib import Path
from loguru import logger
from matplotlib import pyplot as plt
import seaborn as sns


def main(analysis: Analysis):
    top_level_result_dir: str = analysis.top_level_result_dir
    df_dir = top_level_result_dir

    # If results are already obtained; the following block may be commented to only do the result analysis. 
    df = analysis.run_analysis(
        delete_existing_dataframe=True, postprocess_only=False, continue_trials=True
    )

    
    df = analysis.run_analysis(
        delete_existing_dataframe=False, postprocess_only=True, continue_trials=False
    )
    analysis.plots(kwargs_heatmap={"annot": False})
    # print(df.describe())
    # analysis.checks()
    # print(list(df))
    # print(df1.describe())


if __name__ == "__main__":

    # analysis_fine corresponds to the results of the paper. 
    # for analysis_id, analysis in get_analyses().items():
    #     if analysis_id != "analysis_fine":
    #         continue
    #     main(analysis)

    # analysis_coarse runs fewer cases. 
    for analysis_id, analysis in get_analyses().items():
        if analysis_id != "analysis_coarse":
            continue
        main(analysis)