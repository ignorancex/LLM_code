# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=25, collection="Granada", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    gGqf33 = results.gGqf33
    return np.abs(gGqf33)


def inv2(results):
    gGqf33 = results.gGqf33
    gGuf33 = results.gGuf33
    return (gGqf33 * gGuf33) / np.abs(gGqf33)
