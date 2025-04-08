import os
import sys
import torch
import random
import numpy as np
from transformers.file_utils import is_tf_available, is_torch_available
import logging
from sklearn.metrics import accuracy_score

def compute_metrics(pred):
    labels = pred.label_ids3e
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

def setuplogging(args, rank=0):
    root = logging.getLogger()
    if len(root.handlers)<=1:
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(f"[{rank}] [%(levelname)s %(asctime)s] %(message)s")
        handler.setFormatter(formatter)
        root.addHandler(handler)

        fh = logging.FileHandler(os.path.join(args.log_dir,'logging_file.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        root.addHandler(fh)

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)

def to_cuda(data):
    for key in data:
        data[key] = data[key].cuda()
    return data