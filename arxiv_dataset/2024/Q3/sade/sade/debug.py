from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch

import torch.nn as nn
import numpy as np
import tqdm
import io
import inspect

sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils
from models import ncsnpp3d
from models import layerspp
from models import layers
from torchinfo import summary
from configs.ve import toy_config as configs

config = configs.get_config()

# checkpoint = torch.load('exp/ddpm_continuous_vp.pth')

score_model = ncsnpp3d.SegResNetpp(config)
# score_model.load_state_dict(checkpoint)
score_model = score_model.eval()
x = torch.ones(8, 2, *config.data.image_size)
y = torch.tensor([1] * 8)
summary(score_model, input_data=(x, y))
# breakpoint()
with torch.no_grad():
    score = score_model(x, y)
print("Model output of shape:", score.shape)
