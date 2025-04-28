# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=27, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")  # in TeV


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [], check_constrain)
