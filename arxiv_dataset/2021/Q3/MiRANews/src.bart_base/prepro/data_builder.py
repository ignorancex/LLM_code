import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import copy
import torch
import subprocess
from collections import Counter
from os.path import join as pjoin
from prepro.data_train import PreproTrainJson, PreproTrainData


def pretrain_to_json(args):
    obj = PreproTrainJson(args)
    obj.preprocess()

def pretrain_to_data(args):
    obj = PreproTrainData(args)
    obj.preprocess()
