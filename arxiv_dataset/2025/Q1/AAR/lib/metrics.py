import torch
import numpy as np
from sklearn.metrics import average_precision_score

def mean_recall_at_k(preds, gts, k, gt_stats, pred_stats):
    recall = []
    gts = torch.Tensor(gts)
    preds = torch.Tensor(preds)
    for j in range(preds.shape[0]):
        gt = gts[j]                        
        pred = preds[j]        
        gtclasses = (gt.squeeze() == 1).nonzero().squeeze()         
        if gtclasses.ndim == 0:
            gtclasses = [gtclasses.item()]
        v,i = pred.squeeze().sort(descending=True)
        top_pred = i[0:k]                
        ints = np.intersect1d(gtclasses, top_pred)         
        for ci in ints:
            pred_stats[ci] = pred_stats[ci] + 1.
        for ci in gtclasses:
            gt_stats[ci] = gt_stats[ci] + 1.
        if ints != [] :
            recall_k = ints.shape[0]/gtclasses.shape[0]
        else:
            recall_k = 0.
        recall.append(recall_k)
    return recall

def n_recall_at_k(preds, gts, k, data):
    # get n actions of every sample
    length_of_actions = []

    for frame_actions in data:
        length_of_actions.append(len(frame_actions))

    # determine frequency distribution of actions ie., each snapshot consist of 0 to N actions
    freq  = (np.bincount(length_of_actions))

    act_idx = [[] for n in range(len(freq))]

    # store n actions to n idx
    for idx, item in enumerate(data):
        for length in range(len(freq)):
            if len(item) == length:
                act_idx[length].append(idx) 

    # separate the data samples into their respective group of N actions freq = 27 including action length = 0
    gt_n_groups = [None for n in range(len(freq))]
    pred_n_groups = [None for n in range(len(freq))]

    # overall recall for individual length of actions
    n_overall_recall = []
    recall = [] # sanity check 

    for idx, item in enumerate(gt_n_groups):
        # gather the gts and preds based on their respective length of actions and store them using their index
        gt_n_groups[idx] = gts[act_idx[idx]]
        pred_n_groups[idx] = preds[act_idx[idx]]

        recall_n_groups = []

        # if not empty, then compute based on the precision score for the group of 1 to n actions
        if len(gt_n_groups[idx]):
            for j in range(len(gt_n_groups[idx])):
                gt = torch.Tensor(gt_n_groups[idx][j])
                gtclasses = (gt.squeeze() == 1.0).nonzero().squeeze()

                if gtclasses.ndim == 0:
                    gtclasses = np.array([gtclasses.item()])

                pred = torch.Tensor(pred_n_groups[idx][j])
                v, i = pred.squeeze().sort(descending=True)
                top_pred = i[0:k]

                ints = np.intersect1d(gtclasses, top_pred) 

                if ints != []:
                    recall_k = ints.shape[0] / gtclasses.shape[0]
                else:
                    recall_k = 0.0

                recall_n_groups.append(recall_k)
                recall.append(recall_k) # sanity check
            n_overall_recall.append(np.mean(recall_n_groups))

        else: 
            n_overall_recall.append(0.0)

    n_overall_recall = n_overall_recall[1:] # ignore length=0 actions

    return n_overall_recall