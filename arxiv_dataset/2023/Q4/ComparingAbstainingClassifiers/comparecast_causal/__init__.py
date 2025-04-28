"""
comparecast_causal:
    a python package for counterfactually comparing abstaining classifiers
"""

# for convenient imports (`import comparecast_causal as c3`)
from comparecast_causal.abstaining_predictors import *
from comparecast_causal.drconfint import *
from comparecast_causal.experiments import *
from comparecast_causal.learners import *
from comparecast_causal.scoring import *
from comparecast_causal.utils import *

# data submodules (load as, e.g., `ccb.data_utils.synthetic.get_data()`)
from comparecast_causal import data_utils
import comparecast_causal.data_utils.abstaining_classifiers
