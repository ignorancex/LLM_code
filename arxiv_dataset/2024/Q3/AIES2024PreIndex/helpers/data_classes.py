import sys
import warnings

import torch
import torch.nn as nn
import torchvision
import numpy as np

warnings.filterwarnings('ignore')
sys.path.append("helpers")
sys.path.append("models")

def class_count(trainset):
    try:
        unique_labels = list(set(data[1] for data in trainset))
        return unique_labels, len(unique_labels)
    except:
        pass
    try:
        unique_labels = list(set(trainset.labels))
        return unique_labels, len(unique_labels)
    except:
        pass
    try:
        unique_labels = list(set(trainset.classes))
        return unique_labels, len(unique_labels)
    except:
        pass
    