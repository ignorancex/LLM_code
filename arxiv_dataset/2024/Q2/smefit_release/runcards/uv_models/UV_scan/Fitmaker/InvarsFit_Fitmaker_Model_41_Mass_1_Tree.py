# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=41, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    lamSigmaf1 = results.lamSigmaf1
    return np.abs(lamSigmaf1)
