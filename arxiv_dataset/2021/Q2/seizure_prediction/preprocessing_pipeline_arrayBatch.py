#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:03:46 2019

@author: remy.benmessaoud
"""

""" 
goal: make a script that does the preprocessing for all subjects and saves the formatted data.
This could take  more than 7 days
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
"""

#import matplotlib.pyplot as plt
#mport mne
from my_functions import mypreprocessing
import numpy as np 
import os.path
#import matplotlib.pyplot as plt
import os
import pickle
clearFiles = list()
#clearFiles.append( np.array([9 ]))#1
#clearFiles.append( np.array([18 , 19 ]))#2
#clearFiles.append( np.array([5 , 33]))#3
#clearFiles.append( np.array([ ]))#4
#clearFiles.append( np.array([ 15 ]))#5
#clearFiles.append( np.array([ ]))#6
#clearFiles.append( np.array([  ]))#7
#clearFiles.append( np.array([  ]))#8
#clearFiles.append( np.array([  ]))#9
#clearFiles.append( np.array([  ]))#10
#clearFiles.append( np.array([ 30 ]))#11
#clearFiles.append( None )#12
#clearFiles.append( np.array([ 29]))#13
#clearFiles.append( np.array([  6 ]))#14
#clearFiles.append( np.array([ 16 ]))#15
#clearFiles.append( np.array([  15 ]))#16
#clearFiles.append( np.array([3 ]))#17
#clearFiles.append( np.array([ 34]))#18
#clearFiles.append( np.array([ 25 ]))#19
#clearFiles.append( np.array([ 27 ]))#20
#clearFiles.append( np.array([ 17 ]))#21
#clearFiles.append( np.array([ 16 ]))#22
#clearFiles.append( np.array([  ]))#23
#clearFiles.append( np.array([  ]))#24
""" -1 means do all the clear files  : lot of calculations"""
clearFiles.append( np.array([-1 ]))#1
clearFiles.append( np.array([-1 ]))#2
clearFiles.append( np.array([-1]))#3
clearFiles.append( np.array([-1 ]))#4
clearFiles.append( np.array([-1 ]))#5
clearFiles.append( np.array([-1 ]))#6
clearFiles.append( np.array([-1  ]))#7
clearFiles.append( np.array([-1  ]))#8
clearFiles.append( np.array([-1  ]))#9
clearFiles.append( np.array([-1  ]))#10
clearFiles.append( np.array([-1 ]))#11
clearFiles.append( np.array([-1 ]) )#12
clearFiles.append( np.array([-1]))#13
clearFiles.append( np.array([-1 ]))#14
clearFiles.append( np.array([-1 ]))#15
clearFiles.append( np.array([-1 ]))#16
clearFiles.append( np.array([-1 ]))#17
clearFiles.append( np.array([-1]))#18
clearFiles.append( np.array([-1 ]))#19
clearFiles.append( np.array([-1 ]))#20
clearFiles.append( np.array([-1 ]))#21
clearFiles.append( np.array([-1 ]))#22
clearFiles.append( np.array([-1  ]))#23
clearFiles.append( np.array([-1  ]))#24


winSize= 15
preIctal = 10 # min
saveOpt = True
filtOpt = True

# deal with path
currentPath = os.getcwd()
cDir = os.path.basename(currentPath)
if not(cDir == 'neuroProject'):
    newPath = os.path.join(currentPath , 'neuroProject')
    if os.path.isdir(newPath):
        os.chdir(newPath)
    else:
        raise ValueError(' problem of path can not find neuroPpoject directory \nHere is the current path : '\
                         .format(currentPath))
targetPath = os.path.join(currentPath , "myEEGdata/formatted_data/features_standard21_allTimes_{}s".format(winSize))
#=============================================================================
#
##chb3Files=np.array([2])
substr = os.getenv('SLURM_ARRAY_TASK_ID' , "value does not exist")
print("environment variable = {}".format(substr))

sub=int(substr)
print(sub)

#sub =1
#=============================================================================
files2keep = clearFiles[sub-1]

fileSpec = "subject_{}_windowSize_{}_preIctal_{}_features_reduced.pkl".format(sub , winSize , preIctal)
filePath = os.path.join(targetPath , fileSpec) 
#=============================================================================
featsDictionnary = mypreprocessing.new_main_preprocessing(currentPath , sub , winSize , filtOpt , files2keep , preIctal)

"""here save with pickle"""
if saveOpt:
    f = open(filePath,"wb")
    pickle.dump(featsDictionnary,f)
    f.close()