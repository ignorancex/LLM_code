"""

Script to plot loco scores in a single matplotlib figure

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

configs_all = ['baseline/fp_results/predictive', 'baseline_tta/fp_results/predictive', 'dropout/fp_results/predictive', 'dropout/fp_results/aleatoric', 'dropout/fp_results/epistemic', 'dropout_tta/fp_results/predictive', 'dropout_tta/fp_results/aleatoric', 'dropout_tta/fp_results/epistemic', 'ensemble/fp_results/predictive', 'ensemble/fp_results/aleatoric', 'ensemble/fp_results/epistemic', 'ensemble_tta/fp_results/predictive', 'ensemble_tta/fp_results/aleatoric', 'ensemble_tta/fp_results/epistemic']

titles_all = ['Baseline', 'Baseline+TTA', 'Dropout-Predictive', 'Dropout-Aleatoric', 'Dropout-Epistemic', 'Dropout+TTA-Predictive', 'Dropout+TTA-Aleatoric', 'Dropout+TTA-Epistemic', 'Ensemble-Predictive', 'Ensemble-Aleatoric', 'Ensemble-Epistemic', 'Ensemble+TTA-Predictive', 'Ensemble+TTA-Aleatoric', 'Ensemble+TTA-Epistemic']

configs_shape = ['baseline/fp_results/shape', 'baseline_tta/fp_results/shape', 'dropout/fp_results/shape', 'dropout_tta/fp_results/shape', 'ensemble/fp_results/shape', 'ensemble_tta/fp_results/shape']

configs_intensity = ['baseline/fp_results/intensity', 'baseline_tta/fp_results/intensity', 'dropout/fp_results/intensity', 'dropout_tta/fp_results/intensity', 'ensemble/fp_results/intensity', 'ensemble_tta/fp_results/intensity']

titles = ['Baseline', 'Baseline+TTA', 'Dropout', 'Dropout+TTA', 'Ensemble', 'Ensemble+TTA']

#plt.rc('axes', titlesize=20) #fontsize of the title
#plt.rc('axes', labelsize=16) #fontsize of the x and y labels
#plt.rc('xtick', labelsize=12) #fontsize of the x tick labels
#plt.rc('ytick', labelsize=16) #fontsize of the y tick labels


def create_loco_plot(res_dir=None, configs=None, plot_titles=None):

    assert(len(configs) == len(plot_titles))

    if len(configs) == 14:
        n_rows = 4
        n_cols = 4
    elif len(configs) == 6:
        n_rows = 2
        n_cols = 3

    fig, ax = plt.subplots(n_rows, n_cols,figsize=(15, 38))

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            config_idx = row_idx*n_rows + col_idx
            if config_idx == len(configs):
                break
            loco_dir = os.path.join(res_dir, configs[config_idx])
            ordered_loco_means = np.load(os.path.join(loco_dir, 'ordered_loco_means.npy'),
                                         allow_pickle=True)

            ordered_feature_names = np.load(os.path.join(loco_dir, 'ordered_feature_names.npy'),
                                             allow_pickle=True)

            ax[row_idx, col_idx].boxplot(x=ordered_loco_means,
                                        vert=True)

            # Use acronyms for large features names
            labels = []
            for feat_name in ordered_feature_names:
                if len(feat_name) <= 10:
                    labels.append(feat_name)
                else:
                    acronym = ''.join([c for c in feat_name if c.isupper()])
                    if len(acronym) > 1:
                        labels.append(acronym)
                    else:
                        labels.append(feat_name)

            ax[row_idx, col_idx].set_xticklabels(labels,
                                                 rotation=60)

            ax[row_idx, col_idx].set_xlabel('LOCO')
            ax[row_idx, col_idx].set_title(plot_titles[config_idx])
            ax[row_idx, col_idx].set_ylim((-0.02, 0.1))
            ax[row_idx, col_idx].grid(axis='y')

    if n_rows*n_cols > len(configs):
        for empty_ax in ax.flat[len(configs):]:
            empty_ax.remove()

    fig.savefig(os.path.join(res_dir, 'loco_scores.png'),
                bbox_inches='tight')

    fig.savefig(os.path.join(res_dir, 'loco_scores.pdf'),
                bbox_inches='tight')



def create_single_loco_plot(res_dir=None,
                            unc_method='baseline',
                            unc_type='predictive'):

    plt.rc('xtick', labelsize=18) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=18) #fontsize of the y tick labels

    fig, ax = plt.subplots()

    loco_dir = os.path.join(res_dir, unc_method, 'fp_results', unc_type)

    ordered_loco_means = np.load(os.path.join(loco_dir, 'ordered_loco_means.npy'),
                                 allow_pickle=True)

    ordered_feature_names = np.load(os.path.join(loco_dir, 'ordered_feature_names.npy'),
                                     allow_pickle=True)



    ax.boxplot(x=ordered_loco_means,
               labels=ordered_feature_names,
               vert=False)

    ax.set_xlim((-0.02, 0.1))
    ax.grid(axis='x')

    fig.savefig(os.path.join(res_dir, 'loco_scores_{}_{}.png'.format(unc_method, unc_type)),
                bbox_inches='tight')

    fig.savefig(os.path.join(res_dir, 'loco_scores_{}_{}.pdf'.format(unc_method, unc_type)),
                bbox_inches='tight')




if __name__  == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    res_dir = os.path.join(args.root_dir, args.dataset)

    create_loco_plot(res_dir=res_dir,
                     configs=configs_all,
                     plot_titles=titles_all)


    create_single_loco_plot(res_dir=res_dir)
