# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=39, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    lamDelta1f1 = results.lamDelta1f1
    return np.abs(lamDelta1f1)
