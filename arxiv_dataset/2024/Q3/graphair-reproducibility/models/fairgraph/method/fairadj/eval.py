# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import numpy as np
from typing import Sequence, Tuple, List
from scipy import stats
from sklearn.metrics import roc_auc_score,  average_precision_score

THRE = 0.5


def fair_link_eval(
        emb: np.ndarray,
        sensitive: np.ndarray,
        test_edges_true: Sequence[Tuple[int, int]],
        test_edges_false: Sequence[Tuple[int, int]],
        rec_ratio: List[float] = None,
        logger = None,
) -> Sequence[List]:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = np.array(np.dot(emb, emb.T), dtype=np.float128)

    preds_pos = []
    for e in test_edges_true:
        preds_pos.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_neg = []
    for e in test_edges_false:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

    res = {}

    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    acc_score = average_precision_score(labels_all, preds_all)

    res = [acc_score, roc_score]

    standard = res

    return standard