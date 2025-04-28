"""
Script to create visualizations for classification metrics

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
# Set style and font
plt.style.use('seaborn-paper')
mpl.rc("font", family="Serif")

FONTSIZE = 20

configs_all = ['baseline/fp_results/predictive', 'baseline_tta/fp_results/predictive', 'dropout/fp_results/predictive', 'dropout/fp_results/aleatoric', 'dropout/fp_results/epistemic', 'dropout_tta/fp_results/predictive', 'dropout_tta/fp_results/aleatoric', 'dropout_tta/fp_results/epistemic', 'ensemble/fp_results/predictive', 'ensemble/fp_results/aleatoric', 'ensemble/fp_results/epistemic', 'ensemble_tta/fp_results/predictive', 'ensemble_tta/fp_results/aleatoric', 'ensemble_tta/fp_results/epistemic']

configs_shape = ['baseline/fp_results/shape', 'baseline_tta/fp_results/shape', 'dropout/fp_results/shape', 'dropout_tta/fp_results/shape', 'ensemble/fp_results/shape', 'ensemble_tta/fp_results/shape']

configs_intensity = ['baseline/fp_results/intensity', 'baseline_tta/fp_results/intensity', 'dropout/fp_results/intensity', 'dropout_tta/fp_results/intensity', 'ensemble/fp_results/intensity', 'ensemble_tta/fp_results/intensity']

def save_classification_metrics(result_dir=None, combined=False, dataset = None, cross=False, configs=configs_all):

    if combined is True:
        assert(dataset is not None)

    shape_results = False
    intensity_results = False

    if configs[0].split('/')[-1] == 'shape':
        shape_results = True
    elif configs[0].split('/')[-1] == 'intensity':
        intensity_results = True

    # 1. Classification Metrics
    # Merge all the DFs
    res_dfs = []
    for config in configs:
        res_dir = os.path.join(result_dir, config)
        if cross is False:
            if combined is False:
                df = pd.read_csv(os.path.join(res_dir, 'config_metric_df.csv'), index_col=0)
            else:
                df = pd.read_csv(os.path.join(res_dir, 'config_metric_df_{}.csv'.format(dataset)), index_col=0)
        else:
            df = pd.read_csv(os.path.join(res_dir, 'config_metric_df_cross.csv'), index_col=0)

        res_dfs.append(df)

    res_dfs = pd.concat(res_dfs)
    res_dfs = res_dfs.rename(columns = {'Mean Sensitivity' : 'Fraction of FPs detected',
                                        'Mean Specificity': 'Fraction of TPs retained',
                                        'Relative FP Reduction' : 'Relative reduction of false positives',
                                        'Relative TP Retention': 'Relative retention of true positives'})
    print(res_dfs)
    if cross is False:
        if combined is False:
            if shape_results is True:
                res_dfs.to_excel(os.path.join(result_dir, 'dataset_metrics_shape.xlsx'))
                res_dfs.to_csv(os.path.join(result_dir, 'dataset_metrics_shape.csv'))
                res_dfs.to_pickle(os.path.join(result_dir, 'dataset_metrics_shape.pkl'))
            elif intensity_results is True:
                res_dfs.to_excel(os.path.join(result_dir, 'dataset_metrics_intensity.xlsx'))
                res_dfs.to_csv(os.path.join(result_dir, 'dataset_metrics_intensity.csv'))
                res_dfs.to_pickle(os.path.join(result_dir, 'dataset_metrics_intensity.pkl'))
            else:
                res_dfs.to_excel(os.path.join(result_dir, 'dataset_metrics.xlsx'))
                res_dfs.to_csv(os.path.join(result_dir, 'dataset_metrics.csv'))
                res_dfs.to_pickle(os.path.join(result_dir, 'dataset_metrics.pkl'))

        else:
            res_dfs.to_excel(os.path.join(result_dir, 'dataset_metrics_{}.xlsx'.format(dataset)))
            res_dfs.to_csv(os.path.join(result_dir, 'dataset_metrics_{}.csv'.format(dataset)))
            res_dfs.to_pickle(os.path.join(result_dir, 'dataset_metrics_{}.pkl'.format(dataset)))
    else:
        res_dfs.to_excel(os.path.join(result_dir, 'dataset_metrics_cross.xlsx'))
        res_dfs.to_csv(os.path.join(result_dir, 'dataset_metrics_cross.csv'))
        res_dfs.to_pickle(os.path.join(result_dir, 'dataset_metrics_cross.pkl'))

#    errors_df = pd.concat([res_dfs['Error Sensitivity'], res_dfs['Error Specificity'], res_dfs['Error ROC AUC']], axis=1)
#
#    #mpl.rcParams['figure.figsize'] = (20, 14)
#
#    # Plot Bar Graph
#    ax = res_dfs.plot(x="Config",
#                      y=["Fraction of FPs detected", "Fraction of TPs retained", "Mean ROC AUC"],
#                      kind="bar",
#                      yerr=[res_dfs['Error Sensitivity'], res_dfs['Error Specificity'], res_dfs['Error ROC AUC']],
#                      capsize=4,
#                      legend=True)
#
#    ax.set_xlabel("Uncertainty Estimation Methods")
#    ax.set_ylabel("Classification Metrics (Error Bars : 95% CI)")
#    ax.set_yticks(np.arange(0, 1.1, 0.1))
#    ax.legend(loc="lower right")
#    plt.tight_layout()
#    plt.grid(axis="y")
#    if args.cross is False:
#        if combined is False:
#            plt.savefig(os.path.join(result_dir, 'class_metrics.png'))
#            plt.savefig(os.path.join(result_dir, 'class_metrics.pdf'))
#        else:
#            plt.savefig(os.path.join(result_dir, 'class_metrics_{}.png'.format(dataset)))
#            plt.savefig(os.path.join(result_dir, 'class_metrics_{}.pdf'.format(dataset)))
#    else:
#        plt.savefig(os.path.join(result_dir, 'class_metrics_cross.png'))
#        plt.savefig(os.path.join(result_dir, 'class_metrics_cross.pdf'))
#
#    plt.close()
#
#    ax = res_dfs.plot(x="Config",
#                      y=["Relative reduction of false positives", "Relative retention of true positives"],
#                      kind="bar",
#                      yerr=[res_dfs['Error Relative FP Reduction'], res_dfs['Error Relative TP Retention']],
#                      capsize=4,
#                      legend=True)
#
#    ax.set_xlabel("Uncertainty Estimation Methods")
#    ax.set_ylabel("Relative Reduction (Error Bars : 95% CI)")
#    ax.set_yticks(np.arange(0, 1.1, 0.1))
#    ax.legend(loc="lower right")
#    plt.tight_layout()
#    plt.grid(axis="y")
#    if args.cross is False:
#        if combined is False:
#            plt.savefig(os.path.join(result_dir, 'reduction_metrics.png'))
#            plt.savefig(os.path.join(result_dir, 'reduction_metrics.pdf'))
#        else:
#            plt.savefig(os.path.join(result_dir, 'reduction_metrics_{}.png'.format(dataset)))
#            plt.savefig(os.path.join(result_dir, 'reduction_metrics_{}.pdf'.format(dataset)))
#    else:
#        plt.savefig(os.path.join(result_dir, 'reduction_metrics_cross.png'))
#        plt.savefig(os.path.join(result_dir, 'reduction_metrics_cross.pdf'))
#
#    plt.close()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--cross', action='store_true')
    args = parser.parse_args()


    dir_tokens = args.result_dir.split('/')

    if dir_tokens[-1] == '':
        dataset = dir_tokens[-2]
    else:
        dataset = dir_tokens[-1]

#    if dataset == 'combined':
#        save_classification_metrics(result_dir=args.result_dir,
#                                    combined=True,
#                                    dataset='lits',
#                                    cross=False,
#                                    configs=configs_all)
#
#        save_classification_metrics(result_dir=args.result_dir,
#                                    combined=True,
#                                    dataset='umc',
#                                    cross=False,
#                                    configs=configs_all)
#    else:
#        save_classification_metrics(result_dir=args.result_dir,
#                                    combined=False,
#                                    cross=args.cross,
#                                    configs=configs_all)
#
#        if args.cross is False:
#            save_classification_metrics(result_dir=args.result_dir,
#                                        combined=False,
#                                        cross=False,
#                                        configs=configs_shape)

    save_classification_metrics(result_dir=args.result_dir,
                                combined=False,
                                cross=False,
                                configs=configs_intensity)


