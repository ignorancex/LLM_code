# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 03:43:48 2018

@author: 구준모
"""

import pandas as pd
import data_handling as dh
import plotting_deprivation as plt
import matplotlib.pyplot as mplplt
import numpy as np
import csv
import matplotlib.cm as cm



diffusionMap=pd.read_csv('deprivation_score.csv',index_col=0)
eigVecdict=diffusionMap.set_index('GeographyCode').to_dict()

shp,shapeIndex=dh.load_shapefile_with_index('Lower_Layer_Super_Output_Areas_December_2011_Full_Clipped__Boundaries_in_England_and_Wales/Lower_Layer_Super_Output_Areas_December_2011_Full_Clipped__Boundaries_in_England_and_Wales') 
print('1')

plt.Plot_Filled_Map(eigVecdict['IMD_SCORE'], shp, shapeIndex,title='IMD_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('2')

plt.Plot_Filled_Map(eigVecdict['INCOME_SCORE'], shp, shapeIndex,title='Income_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('3')

plt.Plot_Filled_Map(eigVecdict['EMPLOYMENT_SCORE'], shp, shapeIndex,title='Employment_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('4')

plt.Plot_Filled_Map(eigVecdict['HEALTH_DEPRIVATION _AND _DISABILITY_SCORE'], shp, shapeIndex,title='Health_disability_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('5')

plt.Plot_Filled_Map(eigVecdict['EDUCATION_SKILLS_AND_TRAINING_SCORE'], shp, shapeIndex,title='Education_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('6')

plt.Plot_Filled_Map(eigVecdict['BARRIERS_TO_HOUSING_AND_SERVICES_SCORE'], shp, shapeIndex,title='Housing_service_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('7')

plt.Plot_Filled_Map(eigVecdict['CRIME_AND_DISORDER_SCORE'], shp, shapeIndex,title='Crime_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('8')

plt.Plot_Filled_Map(eigVecdict['LIVING_ENVIRONMENT_SCORE'], shp, shapeIndex,title='Living_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('9')

plt.Plot_Filled_Map(eigVecdict['Indoors_Sub-domain_Score'], shp, shapeIndex,title='Indoors_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('10')

plt.Plot_Filled_Map(eigVecdict['Outdoors_Sub-domain_Score'], shp, shapeIndex,title='Outdoors_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('11')

plt.Plot_Filled_Map(eigVecdict['Geographical_Barriers_Sub-domain_Score'], shp, shapeIndex,title='Geographical_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('12')

plt.Plot_Filled_Map(eigVecdict['Wider_Barriers_Sub-domain_Score'], shp, shapeIndex,title='Wider_barriers_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('13')

plt.Plot_Filled_Map(eigVecdict['Children/Young People_Sub-domain_Score'], shp, shapeIndex,title='The_Young_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('14')

plt.Plot_Filled_Map(eigVecdict['Skills_Sub-domain_Score'], shp, shapeIndex,title='Skills_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('15')

plt.Plot_Filled_Map(eigVecdict['IDACI_score'], shp, shapeIndex,title='IDACI_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('16')

plt.Plot_Filled_Map(eigVecdict['IDAOPI_score'], shp, shapeIndex,title='IDAOPI_2010',cmap=cm.seismic, saveDirectory='.', Legend=False,boundaries='Bristol')
print('17')


