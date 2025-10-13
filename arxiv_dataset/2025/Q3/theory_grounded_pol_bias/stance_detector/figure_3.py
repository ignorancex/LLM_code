import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from analysis.helper_functions import confidence_intervals

with open('../data/stance_detector_eval/stats.json', 'r') as infile: conf_stats = json.load(infile)
df = pd.read_csv('../data/eval/eval_df.csv', index_col=0)

conf_intervals = {c : {x : [] for x in conf_stats[c].keys()} for c in conf_stats.keys()}

for conf_level in conf_stats.keys():
    for stat in conf_stats[conf_level].keys():
        conf_intervals[conf_level][stat] = confidence_intervals(conf_stats[conf_level][stat])


df = df.rename(columns={'zeroshot_n': 'zero_n', 'zeroshot_f1': 'zero_f1'})


confidence_intervals(conf_stats['0.9']['zero_f1'])


stat_names = ['n', 'f1', 'zero_n', 'zero_f1']
for statistic in stat_names:
    lower, upper = [], []
    for row in df.iterrows():
        lower_error = row[1][statistic] - conf_intervals[str(row[1]['conf'])][statistic][0]
        upper_error = conf_intervals[str(row[1]['conf'])][statistic][1] - row[1][statistic]
        lower.append(lower_error)
        upper.append(upper_error)
    df[f'{statistic}_lower'] = lower
    df[f'{statistic}_upper'] = upper



# Classifier for biased reponses
plt.rcParams['font.size'] = 16
fig = plt.figure(figsize=(6, 8))  
gs = GridSpec(2, 1, hspace=0.4)
ax1 = fig.add_subplot(gs[0, 0])  


ax1.errorbar(x='conf', y='f1', yerr=[df['f1_lower'], df['f1_upper']],  color='darkred', marker='o', data=df)
ax1.errorbar(x='conf', y='zero_f1', yerr=[df['zero_f1_lower'], df['zero_f1_upper']],  color='darkgrey', marker='v',data=df)

ax1.set_xlabel('Confidence')
ax1.set_ylabel('F1 (macro)')
ax1.set_ylim(0, 1)



ax2 = fig.add_subplot(gs[1, 0])
ax2.errorbar(x='conf', y='n', yerr=[df['n_lower'], df['n_upper']], color='darkred', marker='o', data=df)
ax2.errorbar(x='conf', y='zero_n', yerr=[df['zero_n_lower'], df['zero_n_upper']],color='darkgrey', marker='v', data=df)

ax2.set_xlabel('Confidence')
ax2.set_ylabel('Observations')
plt.legend(['Trained', 'Zeroshot'], loc='lower left')
ax2.set_yticks([50, 100, 150, 200, 250])

plt.show()
fig.savefig('../data/eval/classifier_performance.pdf',bbox_inches='tight')


