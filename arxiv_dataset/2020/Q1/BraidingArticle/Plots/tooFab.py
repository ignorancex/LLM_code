#!/usr/bin/python
# Filename: tooFab.py

import numpy as np

##################################################
# Routine to recast 1D vector into 2D array, given corresponding arrays for the indices
# For use e.g. after having read a text file containing datas in columns
def recast2D(i1,i2,arr1D,verbose=1):

	assert isinstance(i1,np.ndarray),'i1 has to be a numpy array'
	assert i1.ndim==1,'i1 has to be 1-dimensional'
	#assert isinstance(i1[0],int),'i1 has to be an array of integers'

	assert isinstance(i2,np.ndarray),'i2 has to be a numpy array'
	assert i2.ndim==1,'i2 has to be 1-dimensional'
	#assert isinstance(i2[0],int),'i2 has to be an array of integers'

	assert isinstance(arr1D,np.ndarray),'arr1D has to be a numpy array'
	assert arr1D.ndim==1,'arr1D has to be 1-dimensional'

	ii1 = np.unique(i1) ; n1 = np.size(ii1)
	ii2 = np.unique(i2) ; n2 = np.size(ii2) 
	n1D     = np.size(arr1D)
	ntofill = min(n1D,n1*n2)

	if not n1D==n1*n2:
		if verbose==1:
			print('Incorrect size for arr1D: ',n1D,' elements instead of ',n1*n2,'. Filling with the first ',ntofill,' elements.')

	assert i1.size>=ntofill,'i1 does not have enough elements'
	assert i2.size>=ntofill,'i2 does not have enough elements'

	arr2D = np.zeros((n1,n2))

	for i in range(ntofill):
		index1 = (ii1==i1[i])
		index2 = (ii2==i2[i])
		arr2D[index1,index2] = arr1D[i]

	return arr2D

##################################################
# Routine to recast 1D vector into 4D array, given corresponding arrays for the indices
# For use e.g. after having read a text file containing datas in columns
def recast4D(i1,i2,i3,i4,arr1D,verbose=1):

	assert isinstance(i1,np.ndarray),'i1 has to be a numpy array'
	assert i1.ndim==1,'i1 has to be 1-dimensional'
	assert isinstance(i2,np.ndarray),'i2 has to be a numpy array'
	assert i2.ndim==1,'i2 has to be 1-dimensional'
	assert isinstance(i3,np.ndarray),'i3 has to be a numpy array'
	assert i3.ndim==1,'i3 has to be 1-dimensional'
	assert isinstance(i4,np.ndarray),'i4 has to be a numpy array'
	assert i4.ndim==1,'i4 has to be 1-dimensional'

	assert isinstance(arr1D,np.ndarray),'arr1D has to be a numpy array'
	assert arr1D.ndim==1,'arr1D has to be 1-dimensional'

	ii1 = np.unique(i1) ; n1 = np.size(ii1)
	ii2 = np.unique(i2) ; n2 = np.size(ii2)
	ii3 = np.unique(i3) ; n3 = np.size(ii3)
	ii4 = np.unique(i4) ; n4 = np.size(ii4)
 
	n1D     = np.size(arr1D)
	prodn   = n1*n2*n3*n4
	ntofill = min(n1D,prodn)

	if not n1D==prodn:
		if verbose==1:
			print('Incorrect size for arr1D: ',n1D,' elements instead of ',prodn,'. Filling with the first ',ntofill,' elements.')

	assert i1.size>=ntofill,'i1 does not have enough elements'
	assert i2.size>=ntofill,'i2 does not have enough elements'
	assert i3.size>=ntofill,'i3 does not have enough elements'
	assert i4.size>=ntofill,'i4 does not have enough elements'

	arr4D   = np.zeros((n1,n2,n3,n4))

	for i in range(ntofill):
		index1 = (ii1==i1[i])
		index2 = (ii2==i2[i])
		index3 = (ii3==i3[i])
		index4 = (ii4==i4[i])
		arr4D[index1,index2,index3,index4] = arr1D[i]

	return arr4D

##################################################
# Routine to normalize a matrix by its diagonal
def NormalizeMatrix(matrix,diag=None):

	assert matrix.ndim==2,'matrix has to be 2-dimensional'
	nx = matrix.shape[0]
	ny = matrix.shape[1]
	assert nx==ny, 'matrix has to be square'

	if type(diag)!=np.ndarray:
		diag = np.diag(matrix)
	else:
		assert diag.ndim==1,'diag has to be 1-dimensional'
		assert diag.shape[0]==nx, 'diag has to have length compatible with matrix'

	NormMat = matrix/np.sqrt(diag*diag[:,None])

	return NormMat

# End of tooFab.py
