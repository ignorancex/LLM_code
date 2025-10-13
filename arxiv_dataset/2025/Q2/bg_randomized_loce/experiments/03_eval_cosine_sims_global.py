import asyncio
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from bg_randomized_loce.utils.eval_util import cos_sim
from bg_randomized_loce.utils.loce_storage_helpers import _CE_METHOD_KEY, all_pkls_to_npz, df_to_csv, df_from_csv, \
    df_cols_equal
from bg_randomized_loce.utils.logging import log_info, init_logger, get_current_git_hash, log_warn


def cos_sim_on_row(row: pd.Series):
    """Calc cosine similarity between row['ce'] and row['ce_baseline']."""
    ce, ce_base = row['ce'], row['ce_baseline']
    if not isinstance(ce, np.ndarray) or not isinstance(ce_base, np.ndarray):
        log_warn(f"Received non-vector value amongst concept embeddings for row {row}")
        return pd.NA
    return cos_sim(ce, ce_base)


from argparse import ArgumentParser
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results-root", type=str, default=f"/workspace/permanent/loce/loce/results")
    parser.add_argument("--npz-dir", default=None)
    parser.add_argument("--force-update", action="store_true", default=False)
    args = parser.parse_args()

    # Get the calculated LoCEs
    RESULTS_ROOT = args.results_root
    FORCE_UPDATE = args.force_update
    NPZ_DIR = args.npz_dir or RESULTS_ROOT
    # To load all CEs right away:
    #CACHE = f'{RESULTS_ROOT}/index.csv'
    #df = asyncio.run(get_all_ces(RESULTS_ROOT, cache=CACHE))

    # Some initial logging
    init_logger(file_name=os.path.join(RESULTS_ROOT, "logs", f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-cosine_sims.log"),
                log_level=logging.INFO)
    log_info(f"Git Commit Hash: {get_current_git_hash()}")

    ## SIMILARITIES
    # GLOBAL
    # Compare Net2Vecs: cosine sims original <-> bg randomized
    global_ce_method_keys: list[_CE_METHOD_KEY] = ['net2vec_proper_bce', 'net2vec']
    baseline_settings = dict(bg_randomizer_key='vanilla', num_bgs_per_ce=1)
    
    all_cosine_sims: list[pd.DataFrame] = []
    for ce_method in global_ce_method_keys:
        log_info(f"Starting {ce_method=}")
        csv_path: str = os.path.join(RESULTS_ROOT, 'cosine_sims', f"{ce_method}.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        if not FORCE_UPDATE and os.path.isfile(csv_path):
            ce_df: pd.DataFrame = df_from_csv(csv_path)
        else:
            # get the meta info and concept embeddings
            ce_infos = asyncio.run(all_pkls_to_npz(results_dir=RESULTS_ROOT, npz_dir=NPZ_DIR, ce_method=ce_method, force_update=FORCE_UPDATE))
            ce_df: pd.DataFrame = pd.DataFrame(
                #get_ce_pkl_paths(RESULTS_ROOT, ce_method=ce_method))
                ce_infos
            )
            # cache results already here:
            df_to_csv(ce_df, csv_path)

        # split according to whether it is the baseline or not
        is_baseline: pd.DataFrame = df_cols_equal(ce_df, baseline_settings)
        ce_df_baseline = ce_df[is_baseline]
        #ce_df = ce_df[~is_baseline]

        equality_check_cols = ['ce_method', 'dataset_key', 'model_key', 'category_id', 'img_id', 'layer']
        include_cols: list[str] = [c for c in ce_df.columns if 'baseline' not in c and c != 'cos_sim']
        ce_df = ce_df[include_cols].set_index(equality_check_cols).join(
            ce_df_baseline[include_cols].set_index(equality_check_cols),
            rsuffix='_baseline',
        ).reset_index()

        # calc cosine similarities
        ce_df.loc[:, 'cos_sim'] = ce_df.apply(cos_sim_on_row, axis=1)
        ce_df = ce_df.drop(columns=[f'{k}_baseline' for k in baseline_settings.keys()])

        # cosine_sims: list[dict] = []
        # for idx, ce_info in ce_df.iterrows():
            
        #     # get matching baseline
        #     equality_check_on = ce_info[['ce_method', 'dataset_key', 'model_key', 'category_id', 'img_id', 'layer']].to_dict() | baseline_settings
        #     ce_baseline_info = ce_df_baseline[df_cols_equal(ce_df_baseline, equality_check_on)]
        #     if len(ce_baseline_info.index) < 1:
        #         log_warn(f"No baseline for CE: {ce_info=}")
        #         continue
        #     assert len(ce_baseline_info.index) == 1, f"Expected 1 but found {len(ce_baseline_info.index)} possibly matching baselines for {ce_info}"
        #     idx_baseline = ce_baseline_info.index[0]
        #     ce_baseline_info: pd.Series = ce_baseline_info.iloc[0]

        #     # get (and cache) ce
        #     ce: np.ndarray = ce_info.ce
        #     if ce is None:
        #         ce_df.loc[idx, 'ce'] = ce = asyncio.run(get_ce(**ce_info.to_dict()))

        #     # get (and cache) ce_baseline
        #     # get the baseline ce
        #     ce_baseline: np.ndarray = ce_baseline_info.ce
        #     if ce_baseline is None:
        #         ce_df_baseline.loc[idx_baseline, 'ce'] = ce = \
        #             asyncio.run(get_ce(**(ce_info.to_dict() | baseline_settings)))

        #     # Calculate and store cosine sim
        #     curr_cos_sim: float = cos_sim(ce, ce_baseline)
        #     cosine_sims += [{**ce_info, 'ce': ce, 'cos_sim': curr_cos_sim}]

        # # fill baseline rows
        # for _, ce_baseline_info in ce_df_baseline.iterrows():
        #     cosine_sims.append({**ce_baseline_info, 'cos_sim': 0})

        # # add to all
        all_cosine_sims.append(ce_df)

        # cache intermediate results
        log_info(f"Cosine sims for {ce_method}: {ce_df['cos_sim'].mean()}±{ce_df['cos_sim'].std()}")
        df_to_csv(ce_df, csv_path)

    df_to_csv(pd.concat(all_cosine_sims),
              os.path.join(RESULTS_ROOT, 'cosine_sims', f"{"+".join(global_ce_method_keys)}.csv")
             )

    # LOCAL
    # Compare mean LoCEs: cosine sims original <-> bg randomized
    # Compare LoCE dists:
