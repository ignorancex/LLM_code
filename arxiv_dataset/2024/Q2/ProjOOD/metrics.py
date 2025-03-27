from __future__ import print_function
import numpy as np

OOD_data = ''
cifar = np.loadtxt('', delimiter=',')
other = np.loadtxt(OOD_data, delimiter=',')


def tpr95(name):
    #
    Y1 = other[:]
    X1 = cifar[:]

    start = np.min(X1)
    end = np.max(X1)
    gap = (end - start) / 100000

    total = 0.0
    tnr = 0.0

    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float64(len(X1))
        error2 = np.sum(np.sum(Y1 <= delta)) / np.float64(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            tnr += error2
            total += 1
    Tnr = tnr / total


    return Tnr

def auroc(name):
    Y1 = other[:]
    X1 = cifar[:]

    start = np.min(X1)
    end = np.max(X1)
    gap = (end - start) / 100000
    aurocBase = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float64(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float64(len(Y1))
        aurocBase += (-fpr+fprTemp)*tpr
        fprTemp = fpr
    aurocBase += fpr * tpr


    return aurocBase
if __name__ == '__main__':
    print(tpr95('ID'))
    print(auroc('ID'))
