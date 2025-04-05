import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from functools import reduce
import torch
from .trainers import class_report


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fix_state_dict(old_state):
    new_dict = {}

    # kill it with fire
    for key, value in old_state.items():
        if 'orig' in key:
            new_key = key.replace('_orig', '')
            new_dict[new_key] = value 

        elif "mask" in key:
            continue
        else:
            new_dict[key] = value

    return new_dict


class df_builder(object):

    def __init__(self, ground_labels, teacher_pred):
        self.pruning_stats = pd.DataFrame()
        self.PIEs = pd.DataFrame()
        self.ground_labels = ground_labels
        self.teacher_pred = teacher_pred
        self.pie_idx = {}

        # find teacher false positive and negatives
        self.conf = confusion_matrix(ground_labels, teacher_pred)
        diag = np.diag(self.conf)
        self.FP_t = self.conf.sum(axis=0) - diag
        self.FN_t = self.conf.sum(axis=1) - diag

    def add_models(self, paths: list, sparsity: int,
                   method: str, student, dataloaders: dict,
                   dataset_sizes: dict, class_names: list):

        disagree = []
        stats = []

        for path in paths:

            model_stats = {}

            ckpt = torch.load(path, map_location=device)
            if 'model' in ckpt:
                ckpt = ckpt['model']
                
            ckpt = fix_state_dict(ckpt)
            student.load_state_dict(ckpt, strict=True)
            _, student_pred = class_report(student, dataloaders, dataset_sizes)
            student_report = classification_report(self.ground_labels,
                                                   student_pred,
                                                   target_names=class_names,
                                                   output_dict=True)

            model_stats['accuracy'] = student_report['accuracy']
            model_stats['recall'] = student_report['macro avg']['recall']
            model_stats['f1-score'] = student_report['macro avg']['f1-score']
            model_stats['sparsity'] = sparsity
            model_stats['method'] = method
            model_stats['full_report'] = student_report

            conf = confusion_matrix(self.ground_labels, student_pred)
            diag = np.diag(conf)
            FP_s = conf.sum(axis=0) - diag
            FN_s = conf.sum(axis=1) - diag
            model_stats['false postives'] = FP_s
            model_stats['false negatives'] = FN_s
            TP_s = diag
            TN_s = conf.sum() - (FP_s + FN_s + TP_s)

            FP_r = FP_s / (FP_s + TN_s)
            FN_r = FN_s / (FN_s + TP_s)
            model_stats['drift'] = (FP_r) * (FN_r) 
            model_stats['FP_r'] = FP_r
            model_stats['FN_r'] = FN_r
            model_stats['Conf'] = conf
            model_stats['drift_mod'] = np.abs(1 - FP_r) * np.abs(1 - FN_r)

            stats.append(model_stats)
            disagree.append(np.where(self.teacher_pred != student_pred))

        pie_idx = reduce(np.intersect1d, disagree)
        pie_stats = {
            'sparsity': sparsity,
            'method': method,
            'PIEs': len(pie_idx)
        }

        if method in pie_idx:
            self.pie_idx[method][sparsity] = pie_idx
        else:
            self.pie_idx[method] = {sparsity: pie_idx}

        self.pruning_stats = self.pruning_stats.append(
                                                stats,
                                                ignore_index=True
                                            )

        self.PIEs = self.PIEs.append(pie_stats, ignore_index=True)
