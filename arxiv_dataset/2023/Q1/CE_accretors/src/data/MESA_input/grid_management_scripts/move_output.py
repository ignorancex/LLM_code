#!/usr/local/bin/python

import sys
import os
import glob
import math
import numpy as np

sys.path.append("/mnt/home/mrenzo/codes/python_stuff/plotFunc/")
from utilsLib import mvFolder


if __name__ == "__main__":
    ROOT = sys.argv[1]
    folders = glob.glob(ROOT + "/*/")
    # print(folders)
    TARGET_ROOT = sys.argv[2]
    for inF in folders:
        # the line below might need to be generalized
        outF = TARGET_ROOT + inF.split("/")[-2]
        # print(inF)
        # print(outF)
        # print("-----------------")
        mvFolder(inF, outF, "xa_central_lower_limit")
