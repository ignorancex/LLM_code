# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=2, collection="Granada", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    kS = results.kS
    return np.abs(kS)
