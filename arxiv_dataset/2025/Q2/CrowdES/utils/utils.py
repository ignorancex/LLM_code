import random
import numpy as np
import torch
from tqdm import tqdm
from joblib import Parallel


def reproducibility_settings(seed: int = 0):
    """Set the random seed for reproducibility
    
    Params:
        seed (int): random seed
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False  # Settings for 3090
    torch.backends.cudnn.allow_tf32 = False  # Settings for 3090


class ProgressParallel(Parallel):
    """A wrapper for joblib.Parallel that adds a progress bar using tqdm"""

    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
