from os import path

from .version import __version__
from sfit_minimizer.sfit_minimize import *
from sfit_minimizer.sfit_classes import *
from sfit_minimizer.mm_funcs import *

MODULE_PATH = path.abspath(__file__)
for i in range(3):
    MODULE_PATH = path.dirname(MODULE_PATH)

path_1 = path.join(MODULE_PATH, 'data')
if path.isdir(path_1):
    DATA_PATH = path_1
else:
    DATA_PATH = path.join(path.dirname(__file__), 'data')
