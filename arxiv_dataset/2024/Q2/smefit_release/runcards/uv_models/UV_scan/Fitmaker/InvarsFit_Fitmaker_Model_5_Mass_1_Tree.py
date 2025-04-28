# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=5, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    cotBeta = results.cotBeta
    return np.abs(cotBeta)


def inv2(results):
    cotBeta = results.cotBeta
    Z6 = results.Z6
    return (cotBeta * Z6) / np.abs(cotBeta)
