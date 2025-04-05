from pprint import pprint

import numpy as np


def f1(p, r):
    return 2 * p * r / (p + r)


lines = open('t.txt', 'r').readlines()
perfs = [eval(x.split('|')[0].split(':')[1]) for x in lines]
times = [eval(x.split('|')[1].split(':')[1]) for x in lines]

perfs = np.array(perfs).reshape((5, 8, 30, 5))
best_trial_indices = np.argmax(perfs[..., 0], axis=2)
max_metric_rows = perfs[
    np.arange(5)[:, None, None],
    np.arange(8)[None, :, None],
    best_trial_indices[:, :, None],
    :
]
swapped = max_metric_rows.swapaxes(0, 1)
result = np.median(swapped, axis=1)

print('\tpd-pf\t\tpd\t\tpf\tprec\tauc')
pprint(result)
