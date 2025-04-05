from collections import Counter
import numpy as np


def Reweighing(X, y, A):
    # X: independent variables (2-d pd.DataFrame)
    # y: the dependent variable (1-d np.array)
    # A: the name of the sensitive attributes (list of string)
    groups_class = {}
    group_weight = {}
    for i in range(len(y)):
        key_class = tuple([X[a][i] for a in A]+[y[i]])
        key = key_class[:-1]
        if key not in group_weight:
            group_weight[key]=0
        group_weight[key]+=1
        if key_class not in groups_class:
            groups_class[key_class]=[]
        groups_class[key_class].append(i)
    class_weight = Counter(y)
    sample_weight = np.array([1.0]*len(y))
    for key in groups_class:
        weight = class_weight[key[-1]]*group_weight[key[:-1]]/len(groups_class[key])
        for i in groups_class[key]:
            sample_weight[i] = weight
    # Rescale the total weights to len(y)
    sample_weight = sample_weight * len(y) / sum(sample_weight)
    return sample_weight





def FairBalance(X, y, A):
    # X: independent variables (2-d pd.DataFrame)
    # y: the dependent variable (1-d np.array)
    # A: the name of the sensitive attributes (list of string)
    groups_class = {}
    group_weight = {}
    for i in range(len(y)):
        key_class = tuple([X[a][i] for a in A] + [y[i]])
        key = key_class[:-1]
        if key not in group_weight:
            group_weight[key] = 0
        group_weight[key] += 1
        if key_class not in groups_class:
            groups_class[key_class] = []
        groups_class[key_class].append(i)
    sample_weight = np.array([1.0]*len(y))
    for key in groups_class:
        weight = group_weight[key[:-1]]/len(groups_class[key])/2
        for i in groups_class[key]:
            sample_weight[i] = weight

    # Rescale the total weights to len(y)
    # sample_weight = sample_weight * len(y) / sum(sample_weight)
    return sample_weight


def FairBalanceVariant(X, y, A):
    # X: independent variables (2-d pd.DataFrame)
    # y: the dependent variable (1-d np.array)
    # A: the name of the sensitive attributes (list of string)
    groups_class = {}
    for i in range(len(y)):
        key_class = tuple([X[a][i] for a in A] + [y[i]])
        if key_class not in groups_class:
            groups_class[key_class] = []
        groups_class[key_class].append(i)
    sample_weight = np.array([1.0]*len(y))
    for key in groups_class:
        weight = 1.0/len(groups_class[key])
        for i in groups_class[key]:
            sample_weight[i] = weight
    # Rescale the total weights to len(y)
    sample_weight = sample_weight * len(y) / sum(sample_weight)
    return sample_weight
