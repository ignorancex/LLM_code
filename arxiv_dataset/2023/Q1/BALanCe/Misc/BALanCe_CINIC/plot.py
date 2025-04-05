import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os

plt.figure(figsize=(16,16))

for path in glob.glob('*.out'):
    basename = os.path.basename(path)
    basename = basename[:-4]
    array = basename.split('_')
    batch_size=1
    for item in array:
        sub_array = item.split('-')
        if sub_array[0] == 'B':
            batch_size = int(sub_array[1])

    result_lt  = np.loadtxt(path, delimiter=',')
    result_mean = np.mean(result_lt, axis=0)
    result_std = np.std(result_lt, axis=0)
    index_lt = [i*batch_size + 200 for i in range(result_mean.shape[0])]

    plt.plot(index_lt, result_mean, label=basename)
    #plt.fill_between(index_lt, (result_mean-result_std), (result_mean+result_std), alpha=.1, label=basename)
    plt.fill_between(index_lt, (result_mean-result_std), (result_mean+result_std), alpha=.1)

plt.legend()

plt.savefig('learning_curve.png')


