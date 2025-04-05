import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib as mpl

mpl.use("pgf")

## TeX preamble
preamble = [
    r'\usepackage{fontspec}',
#    r'\setmainfont{Linux Libertine O}',
]

params = {
    'font.family': 'serif',
    'text.usetex': True,
    'text.latex.unicode': True,
    'pgf.rcfonts': False,
    'pgf.texsystem': 'xelatex',
    'pgf.preamble': preamble,
}

mpl.rcParams.update(params)

import matplotlib.pyplot as plt
import glob
import os

def get_quartile(out_file):
    basename = os.path.basename(out_file)
    basename = basename[:-4]
    array = basename.split('_')
    batch_size=1
    for item in array:
        sub_array = item.split('-')
        if sub_array[0] == 'B':
            batch_size = int(sub_array[1])

    result_lt  = np.loadtxt(out_file, delimiter=',')
    Q25 = np.percentile(result_lt, 25, axis=0)
    Q50 = np.percentile(result_lt, 50, axis=0)
    Q75 = np.percentile(result_lt, 75, axis=0)

    index_lt = [i*batch_size for i in range(result_lt.shape[1])]

    return index_lt, Q25, Q50, Q75

alpha = 0.2

plt.figure(figsize=(5,5))

index_lt, Q25, Q50, Q75 = get_quartile('Random-batch_B-5_patience-3.out')
plt.plot(index_lt, Q50, 'g', label='Random')
plt.fill_between(index_lt, Q25, Q75, color='g', alpha=alpha)

index_lt, Q25, Q50, Q75 = get_quartile('MIS-MC-Batch_sample-num-10_B-5_M-10000-patience-3.out')
plt.plot(index_lt, Q50, 'b', label='BatchBALD')
plt.fill_between(index_lt, Q25, Q75, color='b', alpha=alpha)

index_lt, Q25, Q50, Q75 = get_quartile('Mean-STD-Batch_sample-num-10_B-5_patience-3.out')
plt.plot(index_lt, Q50, 'c', label='Mean STD')
plt.fill_between(index_lt, Q25, Q75, color='c', alpha=alpha)

index_lt, Q25, Q50, Q75 = get_quartile('Variation-Ratio-Batch_sample-num-10_B-5_patience-3.out')
plt.plot(index_lt, Q50, 'y', label='Variation Ratio')
plt.fill_between(index_lt, Q25, Q75, color='y', alpha=alpha)

index_lt, Q25, Q50, Q75 = get_quartile('ECED-MC-Batch_sample-num-10_hamming-anneal-div4_B-5_M-10000_patience-3.out')
#plt.plot(index_lt, Q50, 'r', label=r'Batch-\textsc{BALanCe} $\tau=\varepsilon/4$')
plt.plot(index_lt, Q50, 'r', label=r'Batch-\textsc{BALanCe}')
plt.fill_between(index_lt, Q25, Q75, color='r', alpha=alpha)

'''
index_lt, Q25, Q50, Q75 = get_quartile('ECED-MC-Batch_sample-num-10_hamming-0.05_B-5_M-10000_patience-3.out')
plt.plot(index_lt, Q50, 'r', label=r'Batch-\textsc{BALanCe} $\tau=0.05$')
plt.fill_between(index_lt, Q25, Q75, color='r', alpha=alpha)

index_lt, Q25, Q50, Q75 = get_quartile('ECED-MC-Batch_sample-num-10_hamming-anneal_B-5_M-10000_patience-3.out')
plt.plot(index_lt, Q50, 'k', label=r'Batch-\textsc{BALanCe} $\tau=valerror/2$')
plt.fill_between(index_lt, Q25, Q75, color='k', alpha=alpha)

index_lt, Q25, Q50, Q75 = get_quartile('ECED-MC-Batch_sample-num-10_hamming-anneal-div4_B-5_M-10000_patience-3.out')
plt.plot(index_lt, Q50, 'm', label=r'Batch-\textsc{BALanCe} $\tau=valerror/2$')
plt.fill_between(index_lt, Q25, Q75, color='m', alpha=alpha)
'''

plt.legend(loc='lower right')
plt.grid()
#plt.xlim(15,275)
plt.ylim(0.0,0.65)

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)

plt.savefig('emnist_byclass_5model_learning_curve.pdf')




