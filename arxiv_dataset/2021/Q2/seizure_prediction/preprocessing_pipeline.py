#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:03:46 2019

@author: remy.benmessaoud
"""

""" 
goal: make a script that does the preprocessing for all subjects and saves the formatted data.
This could take  more than 2 days
"""

#import matplotlib.pyplot as plt
#mport mne
from my_functions import mypreprocessing_intra
import numpy as np 
import os.path
#import matplotlib.pyplot as plt
import os
import pickle

winSize= 15
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
targetPath = os.path.join(currentPath , "myEEGdata/formatted_data/features_intra")
#=============================================================================
#chb3Files=np.array([2, 14 ,17 , 20 , 25 ])
#chb3Files=np.array([2])
clearFiles = np.array([-1])
sub = 2
preIctal = 5 # min
fileSpec = "intra_{}_windowSize_{}_preIctal_{}_features.pkl".format(sub , winSize , preIctal)
filePath = os.path.join(targetPath , fileSpec) 
#=============================================================================
featsDictionnary = mypreprocessing_intra.new_main_preprocessing(currentPath , sub , winSize , filtOpt , clearFiles , preIctal)

"""here save with pickle"""
if saveOpt:
    f = open(filePath,"wb")
    pickle.dump(featsDictionnary,f)
    f.close()

        
        
