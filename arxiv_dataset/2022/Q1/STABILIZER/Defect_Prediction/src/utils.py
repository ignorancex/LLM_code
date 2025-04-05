import os
import sys
from random import shuffle

import pandas as pd
import pickle

import json

def df_norm(dframe, type="normal"):
    """ Normalize a dataframe"""
    col = dframe.columns
    bugs = dframe[dframe.columns[-1]]
    if type == "min_max":
        dframe = (dframe - dframe.min()) / (dframe.max() - dframe.min() + 1e-32)
        dframe[col[-1]] = bugs
        return dframe[col]

    if type == "normal":
        dframe = (dframe - dframe.mean()) / (dframe.std() + + 1e-32)
        dframe[col[-1]] = bugs
        return dframe[col]
