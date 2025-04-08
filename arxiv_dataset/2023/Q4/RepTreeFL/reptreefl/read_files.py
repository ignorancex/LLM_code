from scipy.io import loadmat
import numpy as np
from constants import *

def read_and_preprocess_files():
    source_data = np.random.normal(0, 0.5, (279, 595))
    target_data1 = np.random.normal(0, 0.5, (279, 12720))
    target_data2 = np.random.normal(0, 0.5, (279, 35778))

    return source_data, target_data1, target_data2