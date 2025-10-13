import os
import random

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import numpy as np


def seed_everything(seed: int):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def invert_to_chars(sents, inv_ctc_map):
    sents = sents.detach().numpy()
    outs = []
    for sent in sents:
        for x in sent:
            if x == 0:
                break
            outs.append(inv_ctc_map[x])
    return outs
