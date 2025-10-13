from certifiable_uda_loc.lost_in_the_woods.analysis_liw import (
    get_analyses,
)
from certifiable_uda_loc.lost_in_the_woods.analysis_liw import (
    AnalysisLIW,
)


def main(analysis: AnalysisLIW):

    # If results are already obtained; the following block may be commented to only do the result analysis. 
    df = analysis.run_analysis(
        delete_existing_dataframe=False,
        postprocess_only=False,
        continue_trials=False,
        log_eigval_tightness_thresh=6,
    )

    df = analysis.run_analysis(
        delete_existing_dataframe=False,
        postprocess_only=True,
        continue_trials=False,
        log_eigval_tightness_thresh=6,
    )
    analysis.describe_each_parameter_breakdown(
        columns=[
            "log_eigval_ratio",
            "sdp_error_wrt_mm_gt_init_rot",
            "sdp_error_wrt_mm_gt_init_pos",
            "gt_association_error_rot",
            "gt_association_error_pos",
            "dr_error_wrt_mm_gt_init_rot",
            "dr_error_wrt_mm_gt_init_pos",
            "mm_dr_init_da_wrong_count",
        ]
    )
    analysis.plots(xval="pose_spacing", yval="n_pose", kwargs_heatmap={"annot": True})


if __name__ == "__main__":
    for analysis_id, analysis in get_analyses().items():

        # if analysis_id != "lost_in_the_woods_debug":

        # For Forbes meeting - lost_in_the_woods_by_n_landmarks is good.
        # if analysis_id != "lost_in_the_woods_by_n_landmarks":
        # if analysis_id != "liw_debug":
        if analysis_id != "liw_paper":
            continue
        main(analysis)
