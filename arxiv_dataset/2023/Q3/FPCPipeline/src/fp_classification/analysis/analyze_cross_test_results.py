"""

Script to quantify changes when we perform MR-CT cross-testing

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from scipy.stats import kstest
import os

configs = ['baseline/fp_results/predictive', 'baseline_tta/fp_results/predictive', 'dropout/fp_results/predictive', 'dropout/fp_results/aleatoric', 'dropout/fp_results/epistemic', 'dropout_tta/fp_results/predictive', 'dropout_tta/fp_results/aleatoric', 'dropout_tta/fp_results/epistemic', 'ensemble/fp_results/predictive', 'ensemble/fp_results/aleatoric', 'ensemble/fp_results/epistemic', 'ensemble_tta/fp_results/predictive', 'ensemble_tta/fp_results/aleatoric', 'ensemble_tta/fp_results/epistemic']



def measure_cross_effect(config_dir):
    """
    Measure effect of cross-testing on a single config

    """

    lesion_det_df = pd.read_csv(os.path.join(config_dir, 'config_lesion_metric_df.csv'), index_col=0)
    lesion_det_cross_df = pd.read_csv(os.path.join(config_dir, 'config_lesion_metric_df_cross.csv'), index_col=0)

    effect_dict = {}

    for column in lesion_det_df.columns:
        effect_dict[column] = ((lesion_det_cross_df[column].mean() - lesion_det_df[column].mean())/lesion_det_df[column].mean())*100
        ks_statistic, p_value = kstest(lesion_det_cross_df[column].to_numpy(), lesion_det_df[column].to_numpy())
        effect_dict[column+' (p-value)'] = p_value

    return effect_dict

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True)
    args = parser.parse_args()

    changes_dict = {}
    changes_dict['Config'] = []
    changes_dict['Precision'] = []
    changes_dict['Precision (p-value)'] = []
    changes_dict['Recall'] = []
    changes_dict['Recall (p-value)'] = []
    changes_dict['F1'] = []
    changes_dict['F1 (p-value)'] = []
    for config in configs:
        config_str = config.split('/')[0]
        changes_dict['Config'].append(config_str)
        config_dir = os.path.join(args.result_dir, config)
        effect_dict = measure_cross_effect(config_dir=config_dir)

        for key, value in effect_dict.items():
            changes_dict[key].append(value)

    # Create DF from dict
    changes_df = pd.DataFrame.from_dict(data=changes_dict)
    print(changes_df)
    changes_df.to_csv(os.path.join(args.result_dir, 'cross_testing_analysis.csv'))
    changes_df.to_excel(os.path.join(args.result_dir, 'cross_testing_analysis.xlsx'))



