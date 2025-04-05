from pprint import pprint

import numpy as np


lines = open('t.txt', 'r').readlines()
perfs = [eval(x.split('|')[0].split(':')[1]) for x in lines]
times = [eval(x.split('|')[1]) for x in lines]

perfs = np.array(perfs).reshape((3, 30))
pprint(perfs.max(axis=-1))
