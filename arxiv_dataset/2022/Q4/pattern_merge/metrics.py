import numpy as np
import pandas
import pandas as pd
import math
from apriori_numerical import runApriori
import time
from sklearn.cluster import AgglomerativeClustering
from merge_patterns import merge_patterns
from merge_patterns import merge_patterns_a
import random
from scipy.stats import binom
from scipy.special import betainc
from scipy.special import betaincinv
import matplotlib.pyplot as plt


def construct_distance_matrix(dataset, type):
    similarity_matrix = []
    for _ in dataset.columns:
        similarity_matrix.append([])

    for pointer_1 in range(0, len(dataset.columns)):
        for pointer_2 in range(0, len(dataset.columns)):
            if type == "order-preserving":
                distance = order_preserving(dataset[dataset.columns[pointer_1]], dataset[dataset.columns[pointer_2]])
            else:
                distance = constant(np.array(dataset[dataset.columns[pointer_1]]), np.array(dataset[dataset.columns[pointer_2]]))

            similarity_matrix[pointer_1].append(distance)

    for i in range(0, len(similarity_matrix)):
        for j in range(0, len(similarity_matrix)):
            if i == j:
                similarity_matrix[i][j] = 0

    return similarity_matrix


def constant(v1, v2):
    if len(v1) != len(v2):
        return 0
    unique_pairs = {}

    for i in range(len(v1)):
        if v1[i] == "?" or v2[i] == "?":
            continue
        pair = (v1[i], v2[i])
        if pair in unique_pairs.keys():
            unique_pairs[pair] += 1
        else:
            unique_pairs[pair] = 1

    if len(unique_pairs) == 0:
        return 1

    somation = 0
    u_v1 = {}
    max_entropy = 0
    for val in v1:
        max_entropy += 1/len(v1) * math.log(1/len(v1), 2)
        if val != "?":
            if val not in u_v1.keys():
                u_v1[val] = 1
            else:
                u_v1[val] += 1
    max_entropy *= -1

    u_v2 = {}
    for val in v2:
        if val != "?":
            if val not in u_v2.keys():
                u_v2[val] = 1
            else:
                u_v2[val] += 1

    entropy = 0
    for pair in unique_pairs.keys():
        count_1 = u_v1[pair[0]]
        count_2 = u_v2[pair[1]]
        somation += unique_pairs[pair] / max(count_1, count_2)
        entropy += (unique_pairs[pair]/len(v1)) * math.log(unique_pairs[pair]/len(v1), 2)
    entropy *= -1

    return 1 - ((1 - entropy / max_entropy) * (somation / len(unique_pairs)))


def order_preserving(v1, v2):
    if len(v1) != len(v2):
        return 0

    count_v1_v2 = 0
    count_v2_v1 = 0
    for i in range(len(v1)):
        if v1[i] == np.nan or v2[i] == np.nan:
            continue
        if v1[i] > v2[i]:
            count_v1_v2 += 1
        else:
            count_v2_v1 += 1
    return max(count_v1_v2, count_v2_v1) / len(v1)


'''
a1 = [1, 1, 1, 1, 1, 1, 1]
a2 = [1, 1, 1, 1, 1, 1, 1]
1
print(constant(a1, a2))

a1 = [1, 1, 1, 1, 1, 1, 2]
a2 = [1, 1, 1, 1, 1, 1, 2]
0.78
print(constant(a1, a2))

a1 = [1, 1, 1, 1, 2, 2, 2]
a2 = [1, 1, 1, 1, 2, 2, 2]
0.64
print(constant(a1, a2))

a1 = [1, 1, 1, 2, 2, 3, 3]
a2 = [1, 1, 1, 4, 4, 3, 3]
0.44
print(constant(a1, a2))

a1 = [1, 1, 1, 1, 2, 3, 4]
a2 = [1, 1, 1, 1, 2, 3, 4]
0.40
print(constant(a1, a2))

a1 = [1, 1, 1, 1, 1, 2, 2]
a2 = [1, 1, 1, 2, 2, 3, 3]

print(constant(a1, a2))

a1 = [1, 2, 3, 4, 5, 6, 7]
a2 = [1, 2, 3, 4, 5, 6, 7]

print(constant(a1, a2))



d = pandas.DataFrame(data=[
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 2, 2, 2],
    [1, 1, 1, 1, 2, 2, 2],
    [1, 1, 1, 2, 2, 3, 3],
    [1, 1, 1, 4, 4, 3, 3],
    [1, 1, 1, 1, 2, 3, 4],
    [1, 1, 1, 1, 2, 3, 4],
    [1, 1, 1, 1, 1, 2, 2],
    [1, 1, 1, 2, 2, 3, 3],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7]
]).transpose()

similarity_m = construct_distance_matrix(d, "constant")

'''


