import os
import torch
import random
import importlib
import numpy as np

from datasets.classes_names_and_templates import dataset_info_dict

cudnn_deterministic = True


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def print_summary(acc_taw, acc_tag, forg_taw, forg_tag, weight):
    """Print summary of results"""
    weighted_acc_tag = acc_tag * weight
    weighted_acc_taw = acc_taw * weight
    metric_dict = {'TAw Acc': weighted_acc_taw, 'TAg Acc': weighted_acc_tag, 'TAw Forg': forg_taw, 'TAg Forg': forg_tag}
    weighted = ['TAw Acc', 'TAg Acc']

    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        print('*' * 108)
        print(name)
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.2f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0 and name in weighted:
                    print('\tAvg.:{:5.2f}% '.format(100 * metric_dict[name][i, :i].sum()), end='')
                elif i > 0 and name not in weighted:
                    print('\tAvg.:{:5.2f}% '.format(100 * metric[i, :i].mean()), end='')
            else:
                if name in weighted:
                    print('\tAvg.:{:5.2f}% '.format(100 * metric_dict[name][i, :i + 1].sum()), end='')
                else:
                    print('\tAvg.:{:5.2f}% '.format(100 * metric[i, :i + 1].mean()), end='')
            print()

    avg_over_tasks = []
    for i in range(weighted_acc_tag.shape[0]):
        temp = 100 * weighted_acc_tag[i, :i + 1].sum()
        avg_over_tasks.append(temp)
    print("average accuracy over tasks: ", np.mean(avg_over_tasks))
    print('*' * 108)


def get_dataset_info(dataset):
    if dataset in dataset_info_dict:
        return dataset_info_dict[dataset]['template'], \
               dataset_info_dict[dataset]['classes_names']
    else:
        ex = Exception("unimplemented dataset")
        raise ex


def select_model(classes_names, clip_model, appr):
    CustomCLIP = getattr(importlib.import_module(name='networks.' + appr + '_model'), 'CustomCLIP')
    custom_model = CustomCLIP(classes_names, clip_model)
    if appr == 'l2p' or appr == 'dualprompt':
        OriginalCLIP = getattr(importlib.import_module(name='networks.' + appr + '_model'), 'OriginalCLIP')
        original_model = OriginalCLIP(clip_model)
        return custom_model, original_model
    else:
        return custom_model
