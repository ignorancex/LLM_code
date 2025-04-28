#region Import statements

import importlib
import numpy as np
import math as m
import statistics as st
import cmath
import matplotlib.pyplot as plt
import itertools
import random
from operator import add
from graphviz import Source, render
import pydot

import datetime
import time

import sys
sys.path.append('py_modules') # below modules are in this folder
#print(sys.path)
import setup_nx # your own module, setup.nx.py
importlib.reload(setup_nx)
import my_feeder_funcs as ff
import my_impedance_funcs as imp
import my_configVis_funcs as vis
import my_detControlMatExistence_funcs as ctrl
import my_detLznRange_funcs as lzn
import my_heatmapSetup_funcs as hm

#endregion
#_------------------------------------
# write busnames into a csv
import csv

def write_busLst(v):
    graphNodes_noSub=hm.remove_subst_nodes(v.feeder, v.file_name) # remove substation nodes, busList will have as many entries as R and X matrix length
    assert(len(graphNodes_noSub)*3==len(v.R)) # *3 is because R is 3ph
    with open("123NF_busList.csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='-',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerows(graphNodes_noSub)

def write_RXmat(v):
    # Save R and X matrices to csv to import into matlab
    # np.savetxt reference: https://thispointer.com/how-to-save-numpy-array-to-a-csv-file-using-numpy-savetxt-in-python/
    np.savetxt('Rmat_123NF_accur.csv', v.R, delimiter=',')
    np.savetxt('Xmat_123NF_accur.csv', v.X, delimiter=',')