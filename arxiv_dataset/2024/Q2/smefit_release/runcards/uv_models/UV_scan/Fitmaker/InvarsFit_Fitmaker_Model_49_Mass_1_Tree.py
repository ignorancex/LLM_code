# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=49, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    lamT2f1 = results.lamT2f1
    return np.abs(lamT2f1)
