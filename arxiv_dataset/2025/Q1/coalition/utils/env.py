import random
import torch
import numpy as np
import pytorch_lightning as pl

# Function to set seed
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark
    pl.seed_everything(seed)