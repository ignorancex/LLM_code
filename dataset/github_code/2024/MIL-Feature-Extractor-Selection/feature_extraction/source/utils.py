import os
import random
import numpy as np
import torch

from pathlib import Path

def collate_patch_filepaths(batch):
    item = batch[0]
    idx = torch.LongTensor([item[0]])
    fp = item[1]
    return [idx, fp]