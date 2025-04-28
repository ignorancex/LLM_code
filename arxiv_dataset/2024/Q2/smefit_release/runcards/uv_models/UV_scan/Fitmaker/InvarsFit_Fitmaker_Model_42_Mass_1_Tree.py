# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=42, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    lamSigma1f1 = results.lamSigma1f1
    return np.abs(lamSigma1f1)
