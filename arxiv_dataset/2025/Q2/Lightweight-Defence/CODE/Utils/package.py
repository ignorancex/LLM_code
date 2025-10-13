import os
import sys
import random
import gc
import re
import csv
import copy
import json
import time
import shutil

import inspect

import numpy as np
import pandas as pd

from pprint import pprint

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch
import torch.optim as optim
import torch.nn as nn
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
#import torchattacks
#from torchattacks import FGSM, BIM, PGD

HOME_LOC = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HOME_LOC)


'''
from CODE.Utils.constant import *
from CODE.Utils.constant import UNIVARIATE_DATASET_NAMES as datasets
from CODE.Utils.utils import *

from CODE.train.trainer import Trainer
from CODE.train.classifier import *

from CODE.attack.attacker import Attack
'''