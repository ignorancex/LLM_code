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
    'runs/gru_3L512_wikitext103_word',
    'runs/rnn_3L512_wikitext103_word',
    'runs/lstm_3L512_wikitext103_word',
]

methods = ['FT_BPTT', 'diagonal', 'BPTT']
cell_types = ['simple', 'gru', 'lstm']

type_names = {
    'simple': 'RNN',
    'gru': 'GRU',
    'lstm': 'LSTM',
}

method_styles = {
    'BPTT': (dict(label='BPTT', color='#4285F4', edgecolor='black')),
    'FT_BPTT': dict(label='FT-BPTT', color='#EA4335', edgecolor='black'),
    'diagonal': dict(label='DSF (ours)', color='#FBBC04', edgecolor='black'),
}

width = 0.2

fig, ax = plt.subplots(layout='constrained')

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

        cell_type = config['model']['cell_type']
        method = config['model']['state_transition']

        if method not in methods:
            print(f"\033[93m[Warning] Unknown method {method}, skipping...\033[00m")
            continue
        if cell_type not in cell_types:
            print(f"\033[93m[Warning] Unknown cell type {cell_type}, skipping...\033[00m")
            continue

        best_perplexity = run['valid_perplexity'].min()

        # bar plot, with 3 groups (one per rnn type) and 3 bars per group (one per method)
        print(f"\t->result for {cell_type} {method}: {best_perplexity:.2f}")
        group_idx = cell_types.index(cell_type)
        method_idx = methods.index(method)
        x = group_idx + method_idx * width
        ax.bar(x, best_perplexity, width=width, **method_styles[method]) 
        method_styles[method]['label'] = None  # only show the label once

ax.set_xticks(np.arange(len(cell_types)) + width)
ax.set_xticklabels([type_names[cell_type] for cell_type in cell_types])
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False)

ax.set_ylabel('Perplexity')

ax.legend(loc='upper center', ncol=3)


# ax.grid(axis='y', linestyle='-', alpha=0.5)
# ax.set_axisbelow(True)

fig.savefig('plots/rnn_types_wikitext.png')