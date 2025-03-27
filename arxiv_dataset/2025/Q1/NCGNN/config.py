import torch
import os
import numpy as np
import random


class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_everything(seed):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled=False
