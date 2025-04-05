# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:08:14 2019

@author: 구준모
"""

import pandas as pd
import data_handling as dh
import plotting as plt
import matplotlib.pyplot as mplplt
import numpy as np
import csv
import matplotlib.cm as cm



diffusionMap=pd.read_csv('OA_OAdiff_14.csv',index_col=0)
eigVecdict=diffusionMap.set_index('OA').to_dict()

shp,shapeIndex=dh.load_shapefile_with_index('Output_Area_December_2011_Full_Clipped_Boundaries_in_England_and_Wales/Output_Area_December_2011_Full_Clipped_Boundaries_in_England_and_Wales') 

plt.Plot_Filled_Map(eigVecdict['IMD'], shp, shapeIndex,title='OA_14_2015IMD',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')

