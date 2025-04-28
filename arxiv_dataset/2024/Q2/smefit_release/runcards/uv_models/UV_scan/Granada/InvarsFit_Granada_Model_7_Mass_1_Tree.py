# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=7, collection="Granada", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    kXi1 = results.kXi1
    return np.abs(kXi1)
