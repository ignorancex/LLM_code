import numpy as np
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import glob
import os
import json

def get_quartile(checkpoint):
    batch_size = checkpoint['B']
    legend = checkpoint['method']
    #s_eff_ratio = checkpoint['s_eff_ratio']
    start_size = checkpoint['trial_checkpoint_lt'][0]['starting_size']
    #start_szie = 20
    #result_lt  = np.loadtxt(out_file, delimiter=',')
    print(f'method: {legend}')
    trial_num = len(checkpoint['trial_checkpoint_lt'])
    print(f'trial num: {trial_num}')
    result_lt = []
    for i in range(len(checkpoint['trial_checkpoint_lt'])):
        trial_checkpoint = checkpoint['trial_checkpoint_lt'][i]
        row_result_lt = []
        for j in range(len(trial_checkpoint['test_result_lt'])):
            test_acc = trial_checkpoint['test_result_lt'][j]['test_f1']
            row_result_lt.append(test_acc)
        result_lt.append(row_result_lt)

    result_lt = np.array(result_lt)

    Q25 = np.percentile(result_lt, 25, axis=0)
    Q50 = np.percentile(result_lt, 50, axis=0)
    Q75 = np.percentile(result_lt, 75, axis=0)

    index_lt = [i*batch_size + start_size for i in range(result_lt.shape[1])]

    return legend, index_lt, Q25, Q50, Q75

alpha = 0.2
plt.figure(figsize=(15,15))

method2color = {'Random': 'g', 'Variation-Ratio': 'y', 'Mean-STD': 'c', 
        'Batch-BALanCe': 'r', 'Batch-BALanCe-Lazy': 'm',
        'BatchBALD': 'b', 'BADGE': 'tan', 
        'CoreSet': 'indigo', 'Cluster-Margin': 'olive'}

for path in glob.glob('./*.json'):
    with open(path) as f:
        checkpoint = json.load(f)
        legend, index_lt, Q25, Q50, Q75 = get_quartile(checkpoint)
        color = method2color[legend]
        plt.plot(index_lt, Q50, color, label=f'{legend}')
        plt.fill_between(index_lt, Q25, Q75, color=color, alpha=alpha)

plt.legend(loc='lower right')
plt.grid()
#plt.xlim(25,260)
#plt.ylim(0.7,0.97)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)

plt.savefig('learning_curve_f1.pdf')


