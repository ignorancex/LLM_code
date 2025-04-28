# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=51, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    sRt = results.sRt
    return np.abs(sRt)
