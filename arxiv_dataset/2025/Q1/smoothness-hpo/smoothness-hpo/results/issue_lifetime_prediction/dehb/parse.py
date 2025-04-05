import numpy as np
from pprint import pprint


lines = open('t.txt', 'r').readlines()
perfs = [eval(x.split('|')[0].split(':')[1]) for x in lines]
times = [eval(x.split('|')[1]) for x in lines]

datasets = open('datasets.txt', 'r').readlines()
datasets = [x.split(' ')[1][:-2] for x in datasets]

perfs = np.array(perfs).reshape((3, 30))
print(datasets)
pprint(perfs.max(axis=-1))
