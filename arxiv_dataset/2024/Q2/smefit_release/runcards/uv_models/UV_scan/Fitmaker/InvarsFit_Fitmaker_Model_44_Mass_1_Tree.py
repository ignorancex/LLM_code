# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=44, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    lamDff1 = results.lamDff1
    return np.abs(lamDff1)
