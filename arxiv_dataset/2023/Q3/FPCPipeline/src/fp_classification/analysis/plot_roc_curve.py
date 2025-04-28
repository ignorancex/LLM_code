"""

Script to plot ROC curves for all configurations for a dataset in a single figure

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from argparse import ArgumentParser
from sklearn.metrics import roc_curve
import joblib
from helper_functions import *

# Set style and font
plt.style.use('seaborn-paper')
mpl.rc("font", family="Serif")

configs = ['baseline/fp_results/predictive', 'baseline_tta/fp_results/predictive', 'dropout/fp_results/predictive', 'dropout/fp_results/aleatoric', 'dropout/fp_results/epistemic', 'dropout_tta/fp_results/predictive', 'dropout_tta/fp_results/aleatoric', 'dropout_tta/fp_results/epistemic', 'ensemble/fp_results/predictive', 'ensemble/fp_results/aleatoric', 'ensemble/fp_results/epistemic', 'ensemble_tta/fp_results/predictive', 'ensemble_tta/fp_results/aleatoric', 'ensemble_tta/fp_results/epistemic']

legends = ['Baseline', 'Baseline+TTA', 'MC-Dropout (Predictive)', 'MC-Dropout (Aleatoric)', 'MC-Dropout (Epistemic)', 'MC-Dropout+TTA (Predictive)', 'MC-Dropout+TTA (Aleatoric)', 'MC-Dropout+TTA (Epistemic)', 'Ensemble (Predictive)', 'Ensemble (Aleatoric)', 'Ensemble (Epistemic)', 'Ensemble+TTA (Predictive)', 'Ensemble+TTA (Aleatoric)', 'Ensemble+TTA (Epistemic)']

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True)

    args = parser.parse_args()

    fig = plt.figure()

    cmap = sns.color_palette("tab20")[:len(legends)]

    operating_points = []

    for config, legend, c in zip(configs, legends, cmap):

        config_dir = os.path.join(args.result_dir, config)

        test_df = pd.read_pickle(os.path.join(config_dir, 'test_dataset.pkl'))
        true_labels = test_df['label'].to_numpy()

        # Get all the seed directories
        seed_dirs = [f.path for f in os.scandir(config_dir) if (f.is_dir() and ('seed' in f.name))]

        # Get mean ROC-AUC over all seeds
        metrics_df = pd.read_csv(os.path.join(config_dir, 'config_metric_df.csv'), index_col=0)
        roc_auc = metrics_df['Mean ROC AUC'].to_numpy()[0]
        mean_sensitivity = metrics_df['Mean Sensitivity'].to_numpy()[0]
        mean_specificity = metrics_df['Mean Specificity'].to_numpy()[0]
        pred_scores = []
        for seed_dir in seed_dirs:
            # Load the scores
            pred_scores.append(np.load(os.path.join(seed_dir, 'pred_scores.npy')))

        pred_scores = np.array(pred_scores)
        mean_pred_scores = np.mean(pred_scores, axis=0)

        fp_rate, tp_rate, thresholds = roc_curve(y_true=true_labels,
                                                 y_score=mean_pred_scores,
                                                 pos_label=1)

        # TPR : sensitivity
        # FPR : 1 - specificity
        # Find the operating point
        thresholds_diff = np.abs(thresholds - 0.5)
        min_idx = np.argmin(thresholds_diff)
        print('Operating point threshold : {}'.format(thresholds[min_idx]))

        operating_points.append((fp_rate[min_idx], tp_rate[min_idx]))

        plt.plot(fp_rate, tp_rate, label='{} : {:.3f}'.format(legend, float(roc_auc), c=c))

    # Plot diagonal line to indicate random classification
    random_class = np.arange(0, 1, 0.01)
    plt.plot(random_class, random_class, '--', c='black')

    # Highlight the operating points
    for op_pt in operating_points:
        plt.scatter(op_pt[0], op_pt[1], marker='^', c='black', zorder=4)

    plt.xlabel('True Positives incorrectly classified')
    plt.ylabel('False Positives correctly classified')
    plt.legend(title='Mean ROC-AUC', loc='best', fontsize=7)

    plt.savefig(os.path.join(args.result_dir, 'roc_curve.pdf'))
    plt.close()

