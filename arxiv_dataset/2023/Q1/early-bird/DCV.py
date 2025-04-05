import numpy as np
from scipy.spatial.distance import pdist
import sys


metrics = ['mean','median','min','max','std','instance']

#def euclidean(train_data, test_data):
#
#    return euclidean_distances(train_data,test_data)

def euclidean(data):

    return pdist(data)


def cal_elements(data):

    return [data.mean(), np.median(data), np.min(data), np.max(data), data.std()]


def similarity(train_elements, test_elements):

    sim = {}

    for metric in metrics:
        #print(test_elements/train_elements)

        if 1.6*train_elements[metric] < test_elements[metric]:
            sim[metric] = 3
        elif (1.3*train_elements[metric] < test_elements[metric]) and (test_elements[metric] <= 1.6*train_elements[metric]):
            sim[metric] = 2
        elif (1.1*train_elements[metric] < test_elements[metric]) and (test_elements[metric] <= 1.3*train_elements[metric]):
            sim[metric] = 1
        elif (0.9*train_elements[metric] <= test_elements[metric]) and (test_elements[metric] <= 1.1*train_elements[metric]):
            sim[metric] = 0
        elif (0.7*train_elements[metric] <= test_elements[metric]) and (test_elements[metric] < 0.9*train_elements[metric]):
            sim[metric] = -1
        elif (0.4*train_elements[metric] <= test_elements[metric]) and (test_elements[metric] < 0.7*train_elements[metric]):
            sim[metric] = -2
        elif test_elements[metric] < 0.4*train_elements[metric]:
            sim[metric] = -3
        else:
            print(test_elements[metric])
            print(train_elements[metric])
            # sys.exit()

    return sim


def determine_rule(sim, size_s, size_t):

    if sim['mean']==0 and sim['std']==0:
        return 1
    elif (sim['min']==-3 or sim['min']==3) and (sim['max']==-3 or sim['max']==3) and (sim['instance']==-3 or sim['instance']==3):
        return 2
    elif (sim['std']==3 and size_t<size_s) or (sim['std']==-3 and size_t>size_s):
        return 3
    elif (sim['std']==3 and size_t>size_s) or (sim['std']==-3 and size_t<size_s):
        return 4
    else:
        return 5


def DCV(train_data, test_data):

    train_dist = euclidean(train_data)
    test_dist = euclidean(test_data)

    elements = cal_elements(train_dist)
    elements.append(len(train_data))

    train_elements = {}
    for metric, ele in zip(metrics, elements):
        train_elements[metric] = ele


    elements = cal_elements(test_dist)
    elements.append(len(test_data))

    test_elements = {}
    for metric, ele in zip(metrics, elements):
        test_elements[metric] = ele

    #print(train_elements)
    #print(test_elements)

    sim = similarity(train_elements, test_elements)

    rule = determine_rule(sim,len(train_data),len(test_data))

    return rule

#train = np.array([[1,1,0,0],[0,0,1,1]])
#test = np.array([[1,1,1,0],[1,0,1,1]])
#train = np.array([[1,1,0,0,0],[0,0,1,1,1],[0,0,1,1,1]])
#test = np.array([[1,1,1,0,0],[1,0,1,1,0],[0,0,1,1,1]])

#euclidean(train,test)

#data = np.array([[1,1,0,0,0],[0,0,1,1,1],[1,0,1,1,1]])
#print(euclidean(data))
