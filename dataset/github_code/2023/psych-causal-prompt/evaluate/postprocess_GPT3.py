import numpy as np
import math
from icecream import ic
from datasets import load_from_disk
from collections import Counter
from icecream import ic
import pandas as pd
from tqdm import trange
import sklearn.metrics
from sklearn.metrics import accuracy_score, f1_score

def convert_token_to_label(token):
    tokens_list =[
        ['1', 'one'],
        ['2', 'two'],
        ['3', 'three'],
        ['4', 'four'],
        ['5', 'five'],
    ]
    for i in range(5):
        for j in tokens_list[i]:
            if (j in token):
                return i
    return -1

def generate_logprob_over_five(top_logprobs):
    log_prob = [0,0,0,0,0]
    min_prob = 1e9
    for i in top_logprobs:
        current_label = convert_token_to_label(i)
        if (current_label!=-1):
            log_prob[current_label] += math.e ** top_logprobs[i]
        min_prob = min(top_logprobs[i],min_prob)
    for i in range(5):
        if (log_prob[i] == 0):
            log_prob[i] = min_prob-1
        else:
            log_prob[i] = math.log(log_prob[i])
    return log_prob

def eval_and_offset(train_list, gt_labels):
    # len(result_list) = 20,000, the first 10,000 is train set, used to generate the offset
    # the last 10,000 is the dev set, used to evaluate the performance
    #ic(result_list[0:10])
    #train_list, dev_list = result_list[:len(result_list)//2],result_list[len(result_list)//2:]
    #gt_labels = [i[1] for i in train_list]
    predict_prob = train_list
    predict_labels = predict_prob.argmax(axis=1)

    print("###########before normalize#############")
    ic(Counter(gt_labels), Counter(predict_labels))
    ic(accuracy_score(gt_labels, predict_labels), f1_score(gt_labels, predict_labels, average='weighted'))


    offset = np.array([0.0,0.0,0.0,0.0,0.0])
    lr = 0.01
    cgt = dict(Counter(gt_labels))
    for i in range(10000):
        predict_labels = predict_prob.argmax(axis=1)
        flag = False
        for label in range(5):
            delta = lr*(1-list(predict_labels).count(label)/cgt[label])
            if (delta!=0):
                flag = True
            offset[label]+=delta
            predict_prob[:,label]+=delta
            #ic(offset[label], delta, flag)
        if (not flag):
            break

    predict_labels = predict_prob.argmax(axis=1)
    print("###########after normalize#############")
    ic(Counter(gt_labels), Counter(predict_labels))
    ic(accuracy_score(gt_labels, predict_labels), f1_score(gt_labels, predict_labels, average='weighted'))
    return predict_labels, predict_prob, offset

def apply_offset(test_list, gt_labels, offset):
    print("Now inference the testset")
    predict_prob = np.array(test_list)
    print("###########before normalize#############")
    predict_labels = predict_prob.argmax(axis=1)
    ic(Counter(gt_labels), Counter(predict_labels))
    ic(accuracy_score(gt_labels, predict_labels), f1_score(gt_labels, predict_labels, average='weighted'))
    predict_prob += np.array(offset)
    print("###########after normalize#############")
    predict_labels = predict_prob.argmax(axis=1)
    ic(Counter(gt_labels), Counter(predict_labels))
    ic(accuracy_score(gt_labels, predict_labels), f1_score(gt_labels, predict_labels, average='weighted'))
    return predict_labels, predict_prob

def find_predict_label(file_name, gt_labels, file_name2 = None, file_name3 = None, offset_filename = None):
    gpt3_result = []
    a = np.load(file_name, allow_pickle=True)
    #ic(generate_logprob_over_five({' full': -6.2033577, ' four': -5.498785, ' five': -1.6925747, ' 4': -0.705747, ' 5': -1.1613839}))
    train_list  = []
    for i in a[0]:
        train_list.append(generate_logprob_over_five(i[0]['choices'][0]['logprobs']['top_logprobs'][0]))
    if (file_name2!=None):
        a = np.load(file_name2, allow_pickle=True)
        for i in a[0]:
            train_list.append(generate_logprob_over_five(i[0]['choices'][0]['logprobs']['top_logprobs'][0]))
    if (file_name3!=None):
        a = np.load(file_name3, allow_pickle=True)
        if (len(a)==1):
            a = a[0]
        for i in a:
            train_list.append(generate_logprob_over_five(i[0]['choices'][0]['logprobs']['top_logprobs'][0]))
    test_list = train_list
    gt_labels_train = gt_labels
    if (offset_filename!=None):
        train_list  = []
        gt_labels_train = []
        a = np.load(offset_filename[0], allow_pickle=True)
        for i in a[0]:
            train_list.append(generate_logprob_over_five(i[0]['choices'][0]['logprobs']['top_logprobs'][0]))

        from datasets import load_from_disk
        dataset = load_from_disk(offset_filename[1])
        gt_labels_train = []
        for i in dataset:
            gt_labels_train.append(i['label'])
    #ic(train_list[0:10])
    train_list = np.array(train_list)
    test_list = np.array(test_list)
    gt_labels = np.array(gt_labels)
    ic(len(train_list), len(test_list), len(gt_labels_train),len(gt_labels))
    predict_labels_train, predict_prob_train, offset = eval_and_offset(train_list, gt_labels_train)
    ic(offset)
    predict_labels, predict_prob = apply_offset(test_list, gt_labels, offset)
    return predict_labels, predict_prob


from scipy.special import softmax
def KL(log_px, log_py):
    px = softmax(log_px)
    py = softmax(log_py)
    return np.sum(px*np.log(px/py))

def KL_gt(log_px):
    px = softmax(log_px)
    py = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    #py[label]=1-0.000004
    return np.sum(px*np.log(px/py))

def Entropy(log_px):
    px = softmax(log_px)
    return -np.sum(px*np.log(px))

import torch
from torch import nn
loss = nn.CrossEntropyLoss()
def cross_Entropy(log_px, label):
    log_px = torch.tensor([log_px])
    label = torch.tensor([label])
    l1 = loss(log_px, label)
    return l1.item()


# 好像没有用
def PMI(log_px, log_py):
    pass

if __name__=='__main__':
    dataset = load_from_disk('./yelp_small_test')
    gt_labels = []
    for i in dataset:
        gt_labels.append(i['label'])
    predict_labels1, predict_prob1 = find_predict_label('gpt3_result_setup_1_short_tiny.npy', gt_labels, file_name2 = 'gpt3_result_setup_1_short_small.npy')
    predict_labels2, predict_prob2 = find_predict_label('gpt3_result_setup_2_short_tiny.npy', gt_labels, file_name2 = 'gpt3_result_setup_2_short_small.npy')
    predict_labels3, predict_prob3 = find_predict_label('gpt3_result_setup_3_short_tiny.npy', gt_labels, file_name2 = 'gpt3_result_setup_3_short_small.npy')
    from sklearn.metrics import confusion_matrix
    ic(confusion_matrix(predict_labels1, predict_labels2))
    ic(confusion_matrix(predict_labels1, predict_labels3))
    ic(confusion_matrix(predict_labels2, predict_labels3))
    #"""
    sum=0
    for i in range(len(predict_prob1)):
        max_KL = KL(predict_prob1[i], predict_prob2[i])
        max_KL = max(max_KL, KL(predict_prob1[i], predict_prob3[i]))
        max_KL = max(max_KL, KL(predict_prob2[i], predict_prob3[i]))
        max_KL = max(max_KL, KL(predict_prob2[i], predict_prob1[i]))
        max_KL = max(max_KL, KL(predict_prob3[i], predict_prob1[i]))
        max_KL = max(max_KL, KL(predict_prob3[i], predict_prob2[i]))
        min_KL = KL(predict_prob1[i], predict_prob2[i])
        min_KL = min(max_KL, KL(predict_prob1[i], predict_prob3[i]))
        min_KL = min(max_KL, KL(predict_prob2[i], predict_prob3[i]))

        if (predict_labels1[i]!=predict_labels2[i] or predict_labels2[i]!=predict_labels3[i]):
            continue
            sum+=1
            print(max_KL, dataset[i])
            ic(softmax(predict_prob1[i]),softmax(predict_prob2[i]),softmax(predict_prob3[i]))
            ic(predict_labels1[i],predict_labels2[i],predict_labels3[i], gt_labels[i])
            continue
        if(max_KL>2):
            print(max_KL, dataset[i])
            ic(softmax(predict_prob1[i]),softmax(predict_prob2[i]),softmax(predict_prob3[i]))
            ic(predict_labels1[i],predict_labels2[i],predict_labels3[i], gt_labels[i])

        continue
        if(max_KL<0.001):
            print(max_KL, dataset[i])
            ic(softmax(predict_prob1[i]),softmax(predict_prob2[i]),softmax(predict_prob3[i]))
            ic(predict_labels1[i],predict_labels2[i],predict_labels3[i], gt_labels[i])

    ic(sum)
    #"""

    #KL_dis = []
    #for i in range(len(gt_labels)):
    #    KL_dis.append(KL_gt(predict_prob3[i]))
    #print(KL_dis)