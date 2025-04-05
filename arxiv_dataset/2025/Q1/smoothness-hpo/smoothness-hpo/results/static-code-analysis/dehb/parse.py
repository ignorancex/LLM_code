
import numpy as np


lines = open('t.txt', 'r').readlines()
perfs = [eval(x.split(':')[1]) for x in lines]

datasets = open('datasets.txt', 'r').readlines()

perfs = [np.array(perfs[i:i + 90]) for i in range(0, len(perfs), 90)]
perfs = [x.reshape((3, x.shape[0] // 3, 5)) for x in perfs]
idx = [np.argmax(x[:, :, 0], axis=-1) for x in perfs]
res = [np.median(x[np.arange(x.shape[0]), idx[i], :], axis=0) for i, x in enumerate(perfs)]
res = np.array(res).reshape((8, 5))

print('pd-pf\t\tpd\tpf\t\tprec\tauc')
print(res)
