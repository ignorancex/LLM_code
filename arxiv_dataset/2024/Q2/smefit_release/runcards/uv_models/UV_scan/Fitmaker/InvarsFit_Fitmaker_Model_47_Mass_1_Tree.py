# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=47, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    lamQ7f1 = results.lamQ7f1
    return np.abs(lamQ7f1)
