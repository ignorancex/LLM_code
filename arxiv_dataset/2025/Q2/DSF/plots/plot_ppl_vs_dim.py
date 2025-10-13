import json
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 12,
    "figure.figsize": (6, 4),
    "axes.titlesize": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "legend.frameon": False,
    "legend.loc": "upper right",
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.5,
    "legend.columnspacing": 1.5,
    "legend.borderaxespad": 0.5,
    "legend.borderpad": 0.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "savefig.transparent": False,
    "pdf.fonttype": 42,
    "pdf.compression": 9,
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": r"\usepackage{amsmath}",
    "pgf.rcfonts": False,
})

run_folders = [
    'runs/gru_3L128_wikitext103_word',
    'runs/gru_3L256_wikitext103_word',
    'runs/gru_3L512_wikitext103_word',
    'runs/gru_3L1024_wikitext103_word',
]

methods = ['FT_BPTT', 'diagonal', 'BPTT']

method_styles = {
    'BPTT': (dict(label='BPTT', color='#4285F4')),
    'FT_BPTT': dict(label='Fully Truncated BPTT', color='#EA4335'),
    'diagonal': dict(label='DSF (ours)', color='#FBBC04'),
}


fig, ax = plt.subplots(layout='constrained')

results = {method: {} for method in methods}

for folder in run_folders:
    for run_dir in os.listdir(folder):
        print(f"Processing {folder}/{run_dir}:")

        config_file = os.path.join(folder, run_dir, 'config.json')
        run_file = os.path.join(folder, run_dir, 'training_log.csv')
        if not os.path.exists(config_file) or not os.path.exists(run_file):
            print(f"\033[93m[Warning] Missing files, skipping... \033[00m")
            continue

        config = json.load(open(config_file))
        run = pd.read_csv(run_file)

        method = config['model']['state_transition']
        hidden_dim = config['model']['hidden_size']

        if method not in methods:
            print(f"\033[93m[Warning] Unknown method {method}, skipping...\033[00m")
            continue

        best_perplexity = run['valid_perplexity'].min()

        # bar plot, with 3 groups (one per rnn type) and 3 bars per group (one per method)
        print(f"\t->result for {method} (3L{hidden_dim}): {best_perplexity:.2f}")

        results[method][hidden_dim] = best_perplexity

# plot
for method in methods:
    X, Y = zip(*sorted(results[method].items()))
    ax.plot(X, Y, **method_styles[method]) 
    method_styles[method]['label'] = None  # only show the label once

ax.set_xlabel('Hidden dimension')
ax.set_ylabel('Perplexity')

ax.legend()


fig.savefig('plots/ppl_vs_dim_wikitext.png')