import sys

sys.path.append("/mnt/home/mrenzo/codes/python_stuff/plotFunc/")
from utilsLib import getTerminationCode, check_and_convert
import os
import glob
from joblib import Parallel, delayed


if __name__ == "__main__":
    root = sys.argv[1]
    folders = glob.glob(root + "/*/")
    Parallel(n_jobs=19)(delayed(check_and_convert)(f) for f in folders)
    # for f in folders:
    #     check_and_convert(f)
