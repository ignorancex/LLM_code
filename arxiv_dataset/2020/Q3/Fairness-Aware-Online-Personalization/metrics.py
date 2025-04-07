import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

#############
## Parameters:
##	 k : int which indicates k in Skew@k
##	 top_indices : List of indices to consider
##	 protected_attribute : Numpy array of 0's and 1's as long as number of rows in feature matrix indicating which group the feature belongs to
##	 p0 : (Baseline) Fraction of features belonging to group 0 labelled as positive by an unbiassed user among all features labelled as positive by the user
##	 p1 : (Baseline) Fraction of features belonging to group 0 labelled as positive by an unbiassed user among all features labelled as positive by the user
##
## Return Type: Floating point number representing Skew@k
##
## This function computes skew@k over the chosen features corresponding to indices in top_indices using their group labels
## given in protected_attribute and baseline ratios p0 and p1
def skew(k, top_indices, protected_attribute, p0, p1):
	num = p1
	for i in range(k):
		if(protected_attribute[top_indices[i]] > 0.5):
			num = num + 1
	return np.log(num / (p1 * k))

#############
## Parameters:
##	 top_indices : List of indices to consider
##	 protected_attribute : Numpy array of 0's and 1's as long as number of rows in feature matrix indicating which group the feature belongs to
##	 p0 : (Baseline) Fraction of features belonging to group 0 labelled as positive by an unbiassed user among all features labelled as positive by the user
##	 p1 : (Baseline) Fraction of features belonging to group 0 labelled as positive by an unbiassed user among all features labelled as positive by the user
##
## Return Type: Floating point number representing Skew@k
##
## This function computes NDCS over the chosen features corresponding to indices in top_indices using their group labels
## given in protected_attribute and baseline ratios p0 and p1
def NDCS(top_indices, protected_attribute, p0, p1):
	n = len(top_indices)
	sum = 0.0
	den = 0.0
	for i in range(1,n):
		sum = sum + (1 / np.log2(i + 1)) * skew(i, top_indices, protected_attribute, p0, p1)
		den = den + 1 / np.log2(i + 1)
	sum = sum / den
	return sum

#############
## Parameters:
##	 n : Number of features over which to compute precision
##	 top_indices : List of indices to consider
##	 output : User labels for the feature vectors in the order of feature vectors
##
## Return Type: Floating point number representing precision
##
## This function computes precision that is fraction of feature vectors given by index in top_indices
## that are labelled positive by the user
def precision(n, top_indices, output):
	num = 0.0
	for i in range(n):
		if(output[top_indices[i]] > 0.5):
			num = num + 1.0
	num = num / n
	return num