import os
import argparse
import yaml
import random

import torch
import numpy as np
from sklearn.metrics import classification_report


def makedir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def print_separator(text, total_len=50):
    print('#' * total_len)
    left_width = (total_len - len(text))//2
    right_width = total_len - len(text) - left_width
    print("#" * left_width + text + "#" * right_width)
    print('#' * total_len)


def print_yaml(opt):
    lines = []
    if isinstance(opt, dict):
        for key in opt.keys():
            tmp_lines = print_yaml(opt[key])
            tmp_lines = ["%s.%s" % (key, line) for line in tmp_lines]
            lines += tmp_lines
    else:
        lines = [": " + str(opt)]
    return lines


def create_path(opt):
    for k, v in opt['paths'].items():
        makedir(os.path.join(v, opt['exp_name']))


def read_yaml():
    # read in yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path for the config file")
    parser.add_argument("--exp_ids", type=int, nargs='+', default=[0], help="Path for the config file")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="Path for the config file")
    args = parser.parse_args()

    # torch.cuda.set_device(args.gpu)
    with open(args.config) as f:
        opt = yaml.safe_load(f)
    return opt, args.gpus, args.exp_ids


def should(current_freq, freq):
    return current_freq % freq == 0


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_class_distribution(num_classes, labels):
    target_list = labels[torch.randperm(len(labels))]
    count_dict = {}.fromkeys(np.arange(num_classes), 0)
    for i in labels:
        count_dict[i.data.item()] += 1
    class_count = [i for i in count_dict.values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float32)

    return class_weights[target_list]


def repeat_dataloader(iterable):
    """ repeat dataloader """
    while True:
        for x in iterable:
            yield x


def get_metrics(targets, predicts):
    # targets: [num_tasks, batch_size], predicts: [num_tasks, batch_size, dim]
    if len(predicts.size()) > 2:
        num_tasks, batch_size, dim = predicts.size()
        metrics = {'acc':[], 'macro-f1':[], 'weighted-f1':[]}
        y_softmax = torch.log_softmax(predicts, dim=-1)
        _, y_tags = torch.max(y_softmax, dim=-1) # >> [num_tasks, bsz]
        
        for taskid in range(num_tasks):
            target = targets[taskid].tolist()
            predict = y_tags[taskid].tolist()
            res = classification_report(target, predict, output_dict=True)
            metrics['acc'].append(res['accuracy'])
            metrics['macro-f1'].append(res['macro avg']['f1-score'])
            metrics['weighted-f1'].append(res['weighted avg']['f1-score'])
        
        return metrics
    else:
        batch_size, dim = predicts.size()
        metrics = {'acc':[], 'macro-f1':[], 'weighted-f1':[]}
        y_softmax = torch.log_softmax(predicts, dim=-1)
        _, y_tags = torch.max(y_softmax, dim=-1) # >> [bsz]
        res = classification_report(targets.tolist(), y_tags.tolist(), output_dict=True)
        metrics['acc'].append(res['accuracy'])
        metrics['macro-f1'].append(res['macro avg']['f1-score'])
        metrics['weighted-f1'].append(res['weighted avg']['f1-score'])

        return metrics