# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=46, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    lamQ5f1 = results.lamQ5f1
    return np.abs(lamQ5f1)
