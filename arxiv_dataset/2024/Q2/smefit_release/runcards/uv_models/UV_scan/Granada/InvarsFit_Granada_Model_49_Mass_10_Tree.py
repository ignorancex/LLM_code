# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=49, collection="Granada", mass=10, pto="NLO", eft="NHO")


def inv1(results):
    lamT2f3 = results.lamT2f3
    return np.abs(lamT2f3)
