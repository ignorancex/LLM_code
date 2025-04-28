# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=48, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    lamT1f1 = results.lamT1f1
    return np.abs(lamT1f1)
