from imageability.lib import db

import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy import stats
from sklearn import metrics
from pathlib import Path

parser = argparse.ArgumentParser('baseline_correlation')
parser.add_argument('--input', type=str, help='Path to the baseline measurements', nargs='+')
parser.add_argument('--gt', type=str, help='Path to the word-score pairs')
parser.add_argument('--validation', type=float, help='Ratio of queries to use for validation')
parser.add_argument('--seed', type=int, default=43, help='Seed for the random agenerator')
parser.add_argument("--thresholds", type=float, nargs='+',
                    help="All words below this threshold have low imageability/concreteness")
parser.add_argument("--labels", type=str, nargs='+', help='Plot labels for each input file')
parser.add_argument("--output", type=str, help="Path to the output directory")
parser.add_argument("--prefix", type=str, help="Prefix for figures")
args = parser.parse_args()

if args.output is not None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,bm}'
plt.rcParams.update({'font.size': 36})
colors = ['black', 'blue', 'red', 'darkorange']
linestyles = ['-', '--', '-.', ':']
markers = ['*', 'o', '+', '^']

gt_words, gt_scores = db.load_groundtruth(args.gt)

rng = np.random.default_rng(seed=args.seed)
validation_count = int(args.validation * len(gt_words))
validation_queries_indices = rng.choice(np.arange(len(gt_words)), validation_count)

eval_gt_scores = np.delete(gt_scores, validation_queries_indices, axis=0)
eval_gt_words = []
for i in range(len(gt_words)):
    if i not in set(validation_queries_indices.tolist()):
        eval_gt_words.append(gt_words[i].lower())

thresholds = []
aucs = []

# Load the baseline scores.
for input_file in args.input:
    print(f'Processing {input_file}')
    baseline = {}
    with open(input_file) as f:
        for line in f.readlines():
            parts = line.split(',')
            word, score = parts[0].lower(), parts[1]
            if word in eval_gt_words:
                baseline[word] = float(score)

    # Create ground-truth scores.
    filtered_gt_scores = []
    filtered_predicted_scores = []
    for gt_word, gt_score in zip(eval_gt_words, eval_gt_scores.tolist()):
        if gt_word in baseline:
            filtered_gt_scores.append(gt_score)
            filtered_predicted_scores.append(baseline[gt_word])
    filtered_gt_scores = np.array(filtered_gt_scores)
    filtered_predicted_scores = np.array(filtered_predicted_scores)

    # Compute correlation
    print(f'  Found {len(filtered_gt_scores)} common words ({100 * len(filtered_gt_scores) / float(len(eval_gt_scores)):.2f})')
    print(f'  {stats.spearmanr(filtered_gt_scores, filtered_predicted_scores)}')

    # AUC.
    if args.thresholds is not None:
        _thresholds = []
        _aucs = []
        for threshold in args.thresholds:
            _gt_scores = np.copy(filtered_gt_scores)
            _gt_scores[filtered_gt_scores < threshold] = 2
            _gt_scores[filtered_gt_scores >= threshold] = 1
            fpr, tpr, _ = metrics.roc_curve(_gt_scores, filtered_predicted_scores, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            _thresholds.append(threshold)
            _aucs.append(auc)

        thresholds.append(_thresholds)
        aucs.append(_aucs)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100, constrained_layout=True)    
        for i, width in enumerate(np.linspace(0.35, 0.5, 4)):
            lows = np.arange(filtered_predicted_scores.shape[0])[filtered_predicted_scores <= width]
            highs = np.arange(filtered_predicted_scores.shape[0])[filtered_predicted_scores >= 1 - width]
            if len(lows) == 0 or len(highs) == 0:
                continue
            extreme_set = np.unique(np.concatenate([lows, highs], -1)).reshape(-1)
            extreme_gt_scores = filtered_gt_scores[extreme_set]
            extreme_predicted_scores = filtered_predicted_scores[extreme_set]

            extreme_thresholds = []
            extreme_aucs = []
            for threshold in args.thresholds:
                _gt_scores = np.copy(extreme_gt_scores)
                _gt_scores[extreme_gt_scores < threshold] = 2
                _gt_scores[extreme_gt_scores >= threshold] = 1
                fpr, tpr, _ = metrics.roc_curve(_gt_scores, extreme_predicted_scores, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                extreme_thresholds.append(threshold)
                extreme_aucs.append(auc)

            ax.plot(extreme_thresholds, extreme_aucs, linewidth=4,
                    c=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    markersize=16,
                    linestyle=linestyles[i % len(linestyles)],
                    label=r'$' + f'{width:0.2}' + r'$')

        ax.set_xlabel(r'$\theta$', fontsize=36)
        ax.set_ylabel(r'\textsc{Auc}', fontsize=36)
        plt.legend(fontsize=24)
        plt.savefig(output / f'{args.prefix}_extremes.pdf')
        plt.close()

if len(aucs) > 0:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, constrained_layout=True)
    for i, (theta, auc, label) in enumerate(zip(thresholds, aucs, args.labels)):
        ax.plot(theta, auc, linewidth=4,
                c=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                markersize=16,
                linestyle=linestyles[i % len(linestyles)],
                label=r'\textsc{' + label + '}')

    ax.set_xlabel(r'$\theta$', fontsize=36)
    ax.set_ylabel(r'\textsc{Auc}', fontsize=36)
    plt.legend(fontsize=24)
    plt.savefig(output / f'{args.prefix}_auc.pdf')
    plt.close()
