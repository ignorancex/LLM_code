# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 19:04:26 2019

@author: 구준모
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('top68Bristol2015IMD.csv',
                        header = 0,
                        encoding='utf-8')


true = data["EV_2"] > eigen_thres;
num_true = np.sum(true);

for eigen_thres in np.arange(0, 0.05, 0.000001):
    if num_true == num_true + 1:
        print(eigen_thres)