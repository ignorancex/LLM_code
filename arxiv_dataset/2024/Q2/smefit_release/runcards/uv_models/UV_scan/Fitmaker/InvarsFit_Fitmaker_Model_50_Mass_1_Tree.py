# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=50, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    sLt = results.sLt
    return np.abs(sLt)
