"""
isolated evaluation functions
"""

import numpy as np

from tqdm import tqdm
from models.textual_representation import name_preprocessing


def evaluation(dataloader, hdist, gt, pred, pred_crm):
    
    classnames = dataloader.dataset.class_to_idx
    classnames = name_preprocessing(classnames)
    N_samples = len(gt)

    # overall top1 accuracy
    acc = np.sum(pred==gt) / N_samples
    acc_crm = np.sum(pred_crm==gt) / N_samples
    

    # overall hierarchical distance @ 1
    pred_hdist = np.zeros([N_samples,])
    pred_hdist_crm = np.zeros([N_samples,])
    for i in tqdm(range(N_samples), total=N_samples):
        class_id_gt = gt[i]
        class_id_pred = pred[i]
        class_id_pred_crm = pred_crm[i]

        pred_hdist[i] = hdist[(classnames[class_id_gt], classnames[class_id_pred])]
        pred_hdist_crm[i] = hdist[(classnames[class_id_gt], classnames[class_id_pred_crm])]

    avg_hdist = np.mean(pred_hdist)
    avg_hdist_crm = np.mean(pred_hdist_crm)

    mistake_id = np.where(pred_hdist!=0)[0]
    mistake_id_crm = np.where(pred_hdist_crm!=0)[0]

    norm_mistake = len(mistake_id)
    norm_mistake_crm = len(mistake_id_crm)

    avg_severity = np.sum(pred_hdist[mistake_id]) / norm_mistake
    avg_severity_crm = np.sum(pred_hdist_crm[mistake_id_crm]) / norm_mistake_crm

    print('Base:')
    print('--------------------------------')
    print(f'Top-1: {acc*100:.2f}%')
    print(f'Severity: {avg_severity:.2f}')
    print(f'HieDist@1: {avg_hdist:.2f}')
    print('--------------------------------')
    print('CRM:')
    print('--------------------------------')
    print(f'Top-1: {acc_crm*100:.2f}%')
    print(f'Severity: {avg_severity_crm:.2f}')
    print(f'HieDist@1: {avg_hdist_crm:.2f}')
    print('--------------------------------')

    return 