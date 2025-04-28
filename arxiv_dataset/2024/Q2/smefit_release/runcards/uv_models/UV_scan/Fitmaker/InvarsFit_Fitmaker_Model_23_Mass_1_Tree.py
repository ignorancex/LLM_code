# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=23, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    gWH = results.gWH
    return np.abs(gWH)
