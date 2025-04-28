# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=38, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    lamEff1 = results.lamEff1
    return np.abs(lamEff1)
