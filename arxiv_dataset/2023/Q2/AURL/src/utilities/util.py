import logging
import os.path
import re
import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.autograd import Variable
from src.utilities.config_new import global_config as cfg
import json
from collections import defaultdict

def update_rule_dst(last_mem, new_number):
    new_mem = []
    idx = 0
    for m in last_mem:
        if idx + len(m) >= 9:
            new_mem.append(new_number[idx:])
        else:
            new_mem.append(new_number[idx:idx + len(m)])
        idx += len(m)
    return ','.join(new_mem)


def user_adjacency_pairs(sys_act, user_act, user_slot, bs_vector, t_nlu_v):
    legel = 1
    avg_legel = 0
    n_legel = -1

    user_act = cfg.user_act_id2name[user_act]
    sys_act = cfg.sys_act_id2name[sys_act]

    slot_r = avg_legel
    if 'inform' in user_act:
        if sum(user_slot) > 1: slot_r = n_legel
        if 1 not in user_slot: slot_r = n_legel
        elif bs_vector[user_slot.index(1)] == 1: slot_r = n_legel
        elif 0 in bs_vector[:user_slot.index(1)]: slot_r = n_legel
    elif 'update' in user_act:
        if 1 not in user_slot: slot_r = n_legel
        if 0 in bs_vector and 1 in bs_vector[bs_vector.index(0):]: slot_r = n_legel
        if 2 not in t_nlu_v or user_slot[t_nlu_v.index(2)] != 1: slot_r = n_legel
    elif 'affirm' in user_act:
        if 'confirm' not in sys_act: slot_r = n_legel
    else:
        if user_act != 'restart':
            if sum(user_slot) != 0: slot_r = n_legel

    if sys_act == 'request':
        if 'inform' in user_act:
            return slot_r + legel
    elif 'confirm' in sys_act:
        if 2 in t_nlu_v:
            return legel+slot_r if 'update_' in user_act else n_legel + slot_r
        else:
            if 'inform' in user_act or user_act in ['affirm']:
                return slot_r + legel
    elif sys_act == 'ack':
        if 'inform' in user_act or user_act in ['ack', 'affirm', 'bye']:
            return slot_r + legel
    elif sys_act in ['req_more', 'continue']:
        if 'inform' not in user_act and user_act != 'finish':
            return n_legel + slot_r
    elif sys_act == 'req_correct':
        return slot_r + legel if 'update_' in user_act else slot_r + n_legel
    elif 'signal' in sys_act:
        return slot_r + legel if 'signal' in user_act else slot_r + n_legel
    return slot_r + avg_legel


def ed(s1, s2):
    '''
    最小编辑距离
    :param s1:string
    :param s2:string
    :return:min dist
    '''
    len1 = len(s1)
    len2 = len(s2)
    matrix = [[i + j for j in range(len2 + 1)] for i in range(len1 + 1)]
    for row in range(len1):
        for col in range(len2):
            comp = [matrix[row + 1][col] + 1, matrix[row][col + 1] + 1]
            if s1[row] == s2[col]:
                comp.append(matrix[row][col])
            else:
                comp.append(matrix[row][col] + 1)
            if row > 0 and col > 0:
                if s1[row] == s2[col - 1] and s1[row - 1] == s2[col]:
                    comp.append(matrix[row - 1][col - 1] + 1)
            matrix[row + 1][col + 1] = min(comp)
    return matrix[len1][len2]


def maskedNll(seq, target, pad_id=1):
    """
    Compute the Cross Entropy Loss of ground truth (target) sentence given the model
    S: <START>, E: <END>, W: word token, 1: padding token, P(*): logProb
    Teacher forced logProbs (seq):
        [P(W1) P(W2) P(E) -   -   -]
    Required gtSeq (target):
        [  W1    W2    E  1   1   1]
    Mask (non-zero tokens in target):
        [  1     1     1  0   0   0]

    """
    # Generator a mask of non-padding (non-zero) tokens
    mask = target.data.ne(pad_id)
    loss = 0
    assert isinstance(target, Variable)
    if isinstance(target, Variable):
        mask = Variable(mask, volatile=target.volatile)
    gtLogProbs = torch.gather(seq, 2, target.unsqueeze(2)).squeeze(2)
    maskedNLL = torch.masked_select(gtLogProbs, mask)
    nll_loss = -torch.sum(maskedNLL) / seq.size(1)
    return nll_loss


def compute_f1(predict, target):
    """Compute precision, recall and f1 given a set of gold and prediction items"""

    f1 = f1_score(predict, target, average="micro")
    return f1

def sub_ge(utterance):
    hantonum = {
        '零': 0,
        '一': 1,
        '二': 2,
        '三': 3,
        '四': 4,
        '五': 5,
        '六': 6,
        '七': 7,
        '八': 8,
        '九': 9
    }
    res = re.search('([3-5三四五]个[0-9零一二三四五六七八九])', utterance)
    if res is not None:
        pos_s, pos_e = res.span()
        sub_sen = res.group(0)
        if pos_e + 2 < len(utterance) and utterance[pos_e + 2] in '0123456789':
            utterance = utterance[:pos_e] + utterance[pos_e + 1:]
        if pos_s - 2 >= 0 and utterance[pos_s - 2] in '0123456789':
            utterance = utterance[:pos_s - 1] + utterance[pos_s:]
            pos_s -= 1
            pos_e -= 1
        utterance = utterance[:pos_s] + int(hantonum.get(sub_sen[0], sub_sen[0])) * str(
            hantonum.get(sub_sen[-1], sub_sen[-1])) + utterance[pos_e:]
    return utterance

def sys_adjacency_pairs(used_act, user_act, sys_act, sys_slot, sys_state):
    pos_flag = 1.0
    neg_flag = -1.0
    avg_flag = 0.0

    # for used
    used_r = 0
    # used_r = -(used_act[cfg.sys_act_name2id[sys_act]] - 2) * 0.1

    # for slot
    slot_r = avg_flag
    if sys_act in ['explicit_confirm', 'implicit_confirm', 'compare']:
        if 1 not in sys_slot: slot_r = neg_flag
        if 1 not in sys_slot[:len(sys_state)]: slot_r = neg_flag

    # for act
    act_r = avg_flag
    collected_len = len(''.join(sys_state))
    if sys_act in ['ask_restart', 'other'] or 'signal' in sys_act:
        act_r = neg_flag
    elif sys_act == 'req_correct':
        act_r = pos_flag if user_act == 'deny' else neg_flag
    elif collected_len > 9:
        act_r = pos_flag if sys_act in ['explicit_confirm', 'implicit_confirm'] else neg_flag
        if user_act in ['affirm', 'ack'] and collected_len == 11:
            act_r = pos_flag if sys_act == 'bye' else neg_flag
    if 'inform' in user_act or 'update' in user_act:
        if 'confirm' in sys_act or sys_act == 'ack':
            act_r = pos_flag
    if collected_len <= 9 and user_act in ['affirm', 'ack']:
        act_r = pos_flag if sys_act in ['req_more', 'continue'] else neg_flag

    return used_r + slot_r + act_r


class RunningStat:  # for class AutoNormalization
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        # assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            pre_memo = self._M.copy()
            self._M[...] = pre_memo + (x - pre_memo) / self._n
            self._S[...] = self._S + (x - pre_memo) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)


class AutoNormalization:
    def __init__(self, shape=(), demean=False, destd=True):
        self.demean = demean
        self.destd = destd

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-21)
        return x

    def reset(self):
        pass
