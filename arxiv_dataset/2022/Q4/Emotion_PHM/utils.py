import os
import random
import logging
from pathlib import Path
import shutil
import logging
from functools import wraps

import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch

logger = logging.getLogger(__name__)

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def compute_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(predictions)
    # next, use threshold to turn them into integer predictions
    probs = probs.cpu().numpy()
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels.cpu().numpy()
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    # return as dictionary
    metrics = {"f1": f1_micro_average}
    return metrics


def compute_metrics(predictions, references):
    assert len(predictions) == len(references)

    f1_micro_average = f1_score(y_true=references, y_pred=predictions, average="micro")
    metrics = {"f1": f1_micro_average}
    return metrics


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def log_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} {result.shape}")
        return result

    return wrapper


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model_to_save, tokenizer, global_step, output_dir):
    # delete older checkpoint(s)
    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"checkpoint-*")]

    for checkpoint in glob_checkpoints:
        # logger.info(f"Deleting older checkpoint {checkpoint}")
        shutil.rmtree(checkpoint)

    # Save model checkpoint
    ckpt_output_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_output_dir, exist_ok=True)

    # logger.info(f"Saving model checkpoint to {ckpt_output_dir}")

    model_to_save.save_pretrained(ckpt_output_dir)
    tokenizer.save_pretrained(ckpt_output_dir)
