import numpy as np


def acc_evaluate(true, pred):
    n_true = []
    for i, x in enumerate(true):
        for j, y in enumerate(extract_output(x)):
            if y in extract_output(pred[i]):
                n_true = np.append(n_true, 1)
            else:
                n_true = np.append(n_true, 0)
    return np.mean(n_true)


def abs_evaluate(true, pred):
    n_true = []
    for i, x in enumerate(true):
        if np.array_equal(extract_output(x), extract_output(pred[i])):
            n_true = np.append(n_true, 1)
        else:
            n_true = np.append(n_true, 0)
    return np.mean(n_true)


def extract_output(arr):
    n_arr = []
    for x in arr:
        if 0 < x < 18:
            n_arr = np.append(n_arr, x)
    return n_arr


def h_evaluate(true, pred):
    n_true = []
    n_true2 = []
    n_true3 = []
    n_truex = []
    for i, x in enumerate(true):
        y, y_ = extract_output(x), extract_output(pred[i])
        if np.array_equal(y, y_):
            if len(y) == 1:
                n_true = np.append(n_true, 1)
            elif len(y) == 2:
                n_true2 = np.append(n_true2, 1)
            elif len(y) == 3:
                n_true3 = np.append(n_true3, 1)
            else:
                n_truex = np.append(n_truex, 1)
        else:
            if len(y) == 1:
                n_true = np.append(n_true, 0)
            elif len(y) == 2:
                n_true2 = np.append(n_true2, 0)
            elif len(y) == 3:
                n_true3 = np.append(n_true3, 0)
            else:
                n_truex = np.append(n_truex, 0)
    return [np.mean(n_true), np.mean(n_true2), np.mean(n_true3), np.mean(n_truex)]
