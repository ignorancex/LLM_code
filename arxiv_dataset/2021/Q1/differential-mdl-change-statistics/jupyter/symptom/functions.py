import numpy as np
from scipy.stats import norm

'''
This script provides basic functions for the Gaussian modeling
'''

## Threshold for 0th MDL
def threshod_MDL(delta, d, h):
    return np.log(1/delta) + (1 + delta + d/2) * np.log(h) + np.log(h)
    
## Threshold for 1st MDL
def threshod_F_MDL(delta, d, h):
    return np.log(1/delta) + d * np.log(h/2)

## Threshold for 2nd MDL
def threshod_S_MDL(delta, d, h):
    return 2 * (np.log(1/delta) + d * np.log(h/2))

## Calcuate delta for the 1st D-MDL
def delta_First_NML(score, d, h):
    return np.exp(d * np.log(h/2) - score)

## Calcuate delta for the 2nd D-MDL
def delta_Second_NML(score, d, h):
    return np.exp(d * np.log(h/2) - 0.5 * score)

## Normalizer term for maximum likelihood estimation 
def normalizer_MDL(d, h):
    return d/2 * np.log(h)

## NML codelength within a window at a given cut-off point
def NML_gaussian(window, h):
      
    window_size = len(window)
    window = np.array(window)
    
    parameters_all = norm.fit(window)
    parameters_before = norm.fit(window[:h])
    parameters_after = norm.fit(window[h:])
    
    std_all = parameters_all[1]
    std_before = parameters_before[1]
    std_after = parameters_after[1]

    change_score = window_size * np.log(std_all) - h * np.log(std_before) - (window_size - h) * np.log(std_after) + normalizer_MDL(2, window_size) - normalizer_MDL(2, h) - normalizer_MDL(2, window_size - h)
    
    return change_score


## Maximal NML codelength within a window
def MDL_gaussian_0th(window):
    
    MDL = -10000
    cut_point = -1
    for cut in range(2, len(window)-1):
        score_tmp = NML_gaussian(window, cut)
        if score_tmp > MDL:
            MDL = score_tmp
            cut_point = cut
    return MDL, cut_point
    
## Maximal first-order NML codelength within a window
def MDL_gaussian_1th(window):
    
    MDL = -10000
    cut_point = -1
    for cut in range(2, len(window)-2):
        score_tmp = NML_gaussian(window, cut + 1) - NML_gaussian(window, cut)
        if score_tmp > MDL:
            MDL = score_tmp
            cut_point = cut
    return MDL, cut_point
 
## Maximal second-order NML codelength within a window   
def MDL_gaussian_2nd(window):
    
    MDL = -10000
    cut_point = -1
    for cut in range(3, len(window)-2):
        score_tmp = NML_gaussian(window, cut + 1) - 2 * NML_gaussian(window, cut) + NML_gaussian(window, cut - 1)
        if score_tmp > MDL:
            MDL = score_tmp
            cut_point = cut
    return MDL, cut_point