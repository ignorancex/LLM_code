import pandas as pd
import data_handling as dh
import plotting as plt
import matplotlib.pyplot as mplplt
import numpy as np
import csv
import matplotlib.cm as cm



diffusionMap=pd.read_csv('14LSOA_2015IMD_noneigenOA.csv',index_col=0)
eigVecdict=diffusionMap.set_index('GeographyCode').to_dict()

shp,shapeIndex=dh.load_shapefile_with_index('Lower_Layer_Super_Output_Areas_December_2011_Full_Clipped__Boundaries_in_England_and_Wales/Lower_Layer_Super_Output_Areas_December_2011_Full_Clipped__Boundaries_in_England_and_Wales') 

plt.Plot_Filled_Map(eigVecdict['Eigenvector_2'], shp, shapeIndex,title='falsenegative2015_UK',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='UK')

