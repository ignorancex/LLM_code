import numpy as np
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import glob
import os
import json

def get_quartile(checkpoint_lt):
    def get_result(checkpoint):
        batch_size = checkpoint['B']
        legend = checkpoint['method']
        #s_eff_ratio = checkpoint['s_eff_ratio']
        start_size = checkpoint['trial_checkpoint_lt'][0]['starting_size']
        #start_szie = 20
        #result_lt  = np.loadtxt(out_file, delimiter=',')
        trial_num = len(checkpoint['trial_checkpoint_lt'])
        result_lt = []
        for i in range(len(checkpoint['trial_checkpoint_lt'])):
            trial_checkpoint = checkpoint['trial_checkpoint_lt'][i]
            row_result_lt = []
            for j in range(len(trial_checkpoint['test_result_lt'])):
                test_acc = trial_checkpoint['test_result_lt'][j]['test_f1']
                row_result_lt.append(test_acc)
            result_lt.append(row_result_lt)

        result_lt = np.array(result_lt)
        return legend, batch_size, start_size, result_lt

    result_lt_lt = []
    legend_lt = []
    batch_size_lt = []
    start_size_lt = []
    for checkpoint in checkpoint_lt:
        legend, batch_size, start_size,result_lt = get_result(checkpoint)
        result_lt_lt.append(result_lt)
        legend_lt.append(legend)
        batch_size_lt.append(batch_size)
        start_size_lt.append(start_size)

    legend = legend_lt[0]
    batch_size = batch_size_lt[0]
    start_size = start_size_lt[0] 

    print(f'method: {legend}')
    
    if len(legend_lt) > 1:   
        for i in range(1, len(legend_lt)):
            assert legend_lt[i] == legend
            assert batch_size_lt[i] == batch_size
            assert start_size_lt[i] == start_size

    result_lt = np.concatenate(result_lt_lt, axis=0)
    print(f'trial num: {result_lt.shape[0]}')
    
    Q25 = np.percentile(result_lt, 25, axis=0)
    Q50 = np.percentile(result_lt, 50, axis=0)
    Q75 = np.percentile(result_lt, 75, axis=0)

    index_lt = [i*batch_size + start_size for i in range(result_lt.shape[1])]

    return legend_lt, index_lt, Q25, Q50, Q75

alpha = 0.2
plt.figure(figsize=(5,5))

method2color = {'Random': 'g', 'Variation-Ratio': 'y', 'Mean-STD': 'c', 
        'Batch-BALanCe': 'r', 'BALanCe': 'm',
        'BatchBALD': 'b', 'BALD': 'pink',
        'CoreSet':'purple', 'BADGE': 'lime'}

path_lt = glob.glob('./*.json')
method_set = {}
for path in path_lt:
    basename = os.path.basename(path)
    array = basename.split('_')
    if array[0] not in method_set:
        method_set[array[0]] = 1
    else:
        method_set[array[0]] += 1

for method in method_set:
    checkpoint_lt = []
    for path in glob.glob(f'./{method}_*.json'):
        with open(path) as f:
            checkpoint = json.load(f)
            checkpoint_lt.append(checkpoint)
    
    legend_lt, index_lt, Q25, Q50, Q75 = get_quartile(checkpoint_lt)
    color = method2color[legend_lt[0]]
    plt.plot(index_lt, Q50, color, label=f'{legend_lt[0]}')
    plt.fill_between(index_lt, Q25, Q75, color=color, alpha=alpha)

plt.legend(loc='lower right')
plt.grid()
#plt.xlim(25,260)
#plt.ylim(0.7,0.97)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)

plt.savefig('learning_curve_f1.pdf')


