from pprint import pprint

import numpy as np


def f1(p, r):
    return 2 * p * r / (p + r)


lines = open('t.txt', 'r').readlines()
perfs = [eval(x.split('|')[0].split(':')[1]) for x in lines]
times = [eval(x.split('|')[1].split(':')[1]) for x in lines]

perfs = np.array(perfs).reshape((9, 30, 4))
idx = np.argmax(perfs[:, :, 0], axis=-1)
res = perfs[np.arange(perfs.shape[0]), idx, :]

print('\t\tf1\t\tpd\t\tpf\tprec')
pprint(res)
