import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from model import *
from metrics import *
import sys

#############
## Parameters:
##	 A : Matrix of size m x n
##	 b : Vector of size n
##
## Return Type: Vector A*b of size m.
##
## This function multiplies A and b and returns A*b
def multiply_mat_vec(A, b):
	c = A[:,0] * b[0]
	for i in range(1, len(A[0])):
		c = c + A[:,i] * b[i]
	return c

#############
## Parameters:
##	 k : Number of indices we want from the top
##	 features : 2D numpy array of features. Every row corresponds to one feature vector
##	 wt : Weights vector used to score each feature. Score is the dot product of wt with feature vector
##	 visited: Numpy array of 0's and 1's as long as number of rows in feature matrix. 1 stands for feature vector already visited
##
## Return Type: List of k indices
##
## This function returns the indices of the top scoring k features which have not been visited till now.
## The k indices of visited vector are marked with 1 (meaning visited).
## The visited vector is updated before returning.
def get_top_k(k, features, wt, visited):
	# Faster for small k
	top_indices = [0]*k
	for i in range(k):
		top_indices[i] = get_top_1(features, wt, visited)
	return top_indices



#############
## Parameters:
##	 features : 2D numpy array of features. Every row corresponds to one feature vector
##	 wt : Weights vector used to score each feature. Score is the dot product of wt with feature vector
##	 visited: Numpy array of 0's and 1's as long as number of rows in feature matrix. 1 stands for feature vector already visited
##
## Return Type: List of 1 index
##
## This function returns the index of the top scoring feature which has not been visited till now.
## The index of the visited vector is marked with 1 (meaning visited).
## The visited vector is updated before returning.
def get_top_1(features, wt, visited):
	num_attributes = len(features[0])
	score_vec = multiply_mat_vec(features, wt[:-1])

	# suppress scores of visited points
	score_vec = score_vec - np.absolute(np.max(score_vec)) * visited
	top_indices = np.argmax(score_vec)

	while(True):
		if(visited[top_indices] == 1):
			score_vec[top_indices] = score_vec[top_indices] - np.absolute(np.max(score_vec)) - 1
		else:
			break

		top_indices = np.argmax(score_vec)

	visited[top_indices] = 1
	return top_indices

#############
## Parameters:
##	 f : 2D numpy array of one feature vector.
##   output : The classification label 0 or 1 corresponding to the feature vector
##	 perceptron : Current weights of the perceptron ( perceptron = [w d]. predicted output = Dot-Product(w, f) - d)
##	 lr: Learning Rate
##
## Return Type: None
##
## This function trains the perceptron exactly once on the feature vector f with the given output
## The perceptron weights are updated before returning.
def updatePerceptron(f, output, perceptron, lr):
	extended_f = np.append(f,[-1])
	percep_result = np.dot(perceptron, extended_f)
	percep_result = np.sign(percep_result)
	result = 2*output - 1

	perceptron[:] = perceptron[:] + lr*(result - percep_result)*extended_f[:]

	return

#############
## Parameters:
##	 f : 2D numpy array of one feature vector.
##   output : The classification label 0 or 1 corresponding to the feature vector
##	 perceptron : Current weights of the perceptron ( perceptron = [w d]. predicted output = Dot-Product(w, f) - d)
##	 lr: Learning Rate
##	 reg: Regularization Parameter (lambda) for fairness
##	 w_reg: Regularization weights (learnt from protected_attributes) to ensure fairness
##
## Return Type: None
##
## This function trains the perceptron exactly once on the feature vector f with the given output using the fair regularisation
## The perceptron weights are updated before returning.
def updatePerceptronFair(f, output, perceptron, lr, reg, w_reg):
	extended_f = np.append(f,[-1])
	percep_result = np.dot(perceptron, extended_f)
	percep_result = np.sign(percep_result)
	result = 2*output - 1

	perceptron[:] = perceptron[:] + lr*(result - percep_result)*extended_f[:] - reg*(np.dot(w_reg, perceptron))*w_reg[:]

	return

#############
## Parameters:
##	 f : 2D numpy array of one feature vector.
##   output : The classification label 0 or 1 corresponding to the feature vector
##	 perceptron : Current weights of the perceptron ( perceptron = [w d]. predicted output = Dot-Product(w, f) - d)
##	 lr: Learning Rate
##
## Return Type: None
##
## This function trains the logistic-regression based perceptron exactly once on the feature vector f with the given output
## The perceptron weights are updated before returning.
def updateLR(f, output, perceptron, lr):
	pred = np.dot(perceptron[:-1], f) - perceptron[-1]
	pred = sigmoid(pred)
	perceptron[:-1] = perceptron[:-1] + lr * (output - pred) * f
	perceptron[-1] = perceptron[-1] + lr * (output - pred) * (-1)
	return perceptron

#############
## Parameters:
##	 features : 2D numpy array of features. Every row corresponds to one feature vector
##   output : The classification labels 0 or 1 corresponding to each feature vector
##	 perceptron : Current weights of the perceptron ( perceptron = [w d]. predicted output = Dot-Product(w, f) - d)
##	 lr: Learning Rate
##   num_Rounds_warm: Number of points to train
##
## Return Type: None
##
## This function picks num_Rounds_warm feature vectors uniformly at random trains the perceptron
## exactly once on those feature vectors.
## The perceptron weights are updated before returning.
def constructWarmStartModelOnline(features, output, perceptron, lr, num_Rounds_warm):
	##Warm Start
	for i in range(num_Rounds_warm):
		index = np.random.randint(0, len(features))
		updatePerceptron(features[index], output[index], perceptron, lr)
	perceptron[:] = perceptron[:]/np.linalg.norm(perceptron[:])
	return

#############
## Parameters:
##	 features : 2D numpy array of features. Every row corresponds to one feature vector
##   output : The classification labels 0 or 1 corresponding to each feature vector
##	 perceptron : Current weights of the perceptron ( perceptron = [w d]. predicted output = Dot-Product(w, f) - d)
##	 reg: Regularization Parameter for L2 penalty
##
## Return Type: None
##
## This function solves a Linear Ridge Regression problem between features and output.
## The objective is to match all points lieing on one side of the line to label 0 and the other side to label 1.
def constructWarmStartModelL2Reg(features, output, reg):
	n = -1 * np.ones(len(features))
	n = n.reshape(-1, 1)
	A = np.append(features, n, axis=1)

	b = 2*output - 1
	b = b.reshape(-1, 1)

	w = LinearSysSolverByPsuedoInv(A, b, reg)
	w = w.reshape(-1)

	w[:] = w[:]/np.linalg.norm(w[:])
	return w[:]

#############
## Parameters:
##	 A : m x n Matrix (Numpy Array)
##   b : m x 1 Matrix (Numpy Array)
##	 reg: Regularization Parameter for L2 penalty
##
## Return Type: n x 1 numpy array x, such that Ax = b approximately.
##
## This function solves a Linear System of equations Ax = b using pseudo inverse with L2 penalty on x.
## It tries to minimize ||Ax - b||^2 + reg*||x||^2.
def LinearSysSolverByPsuedoInv(A, b, reg):
	C = np.einsum('ij, jk -> ik', A.T, A) + reg*np.eye(4)
	Cinv = np.linalg.inv(C)
	d = np.einsum('ij, jk -> ik', A.T, b)
	x = np.einsum('ij, jk -> ik', Cinv, d)
	return x[:]

#############
## Parameters:
##	 features : 2D numpy array of features. Every row corresponds to one feature vector
##   output : The classification labels 0 or 1 corresponding to each feature vector
##	 perceptron : Current weights of the perceptron ( perceptron = [w d]. predicted output = Dot-Product(w, f) - d)
##	 lr: Learning Rate
##	 visited: Numpy array of 0's and 1's as long as number of rows in feature matrix. 1 stands for feature vector already visited
##   num_Rounds_online: Number of points to train
##
## Return Type: None
##
## This function picks the top feature scored by the preceptron that is unvisited and trains the perceptron. Every time a feature
## is used for training, its index is marked 1 in visited vector so that it is not chosen again. This is continued num_Rounds_warm
## number of times. A progress bar shows the extent of completion. The perceptron weights are updated before returning.
def onlineTrainingProcess(features, output, perceptron, lr, visited, num_Rounds_online):
	print("Online Learning: ")
	training_points = []
	for i in range(num_Rounds_online):
		candidate = get_top_k(1, features, perceptron, visited)
		index = candidate[0]
		candidate = features[index]
		result = output[index]
		updatePerceptron(candidate, result, perceptron, lr)
		training_points = training_points + [index]

		## Progress Bar
		sys.stdout.write('\r')
		sys.stdout.write("[%-50s] %d%%" % ('=' * ((i+1) * 50 / num_Rounds_online), (i + 1) * 100 / num_Rounds_online ))
		sys.stdout.flush()
	print("")

	perceptron[:] = perceptron[:]/np.linalg.norm(perceptron[:])
	return training_points

#############
## Parameters:
##	 features : 2D numpy array of features. Every row corresponds to one feature vector
##   output : The classification labels 0 or 1 corresponding to each feature vector
##	 perceptron : Current weights of the perceptron ( perceptron = [w d]. predicted output = Dot-Product(w, f) - d)
##	 lr: Learning Rate
##	 visited: Numpy array of 0's and 1's as long as number of rows in feature matrix. 1 stands for feature vector already visited
##   num_Rounds_online: Number of points to train
##   sigma: Covariance matrix of data features
##   reg: Regularization Parameter (lambda) for fairness
##   protected_weights: Weights of a linear model which tries to estimate protected_attribute from feature vectors
##
## Return Type: None
##
## This function picks the top feature scored by the preceptron that is unvisited and trains the perceptron. Every time a feature
## is used for training, its index is marked 1 in visited vector so that it is not chosen again. This is continued num_Rounds_warm
## number of times. A progress bar shows the extent of completion. The perceptron weights are updated using fair regularization before returning.
def onlineTrainingProcessFair(features, output, perceptron, lr, visited, num_Rounds_online, sigma, reg, protected_weights):
	w_reg = multiply_mat_vec(sigma, protected_weights)

	print("Online Learning: ")
	training_points = []
	for i in range(num_Rounds_online):
		candidate = get_top_k(1, features, perceptron, visited)
		index = candidate[0]
		candidate = features[index]
		result = output[index]
		updatePerceptronFair(candidate, result, perceptron, lr, reg, w_reg)
		training_points = training_points + [index]

		## Progress Bar
		sys.stdout.write('\r')
		sys.stdout.write("[%-50s] %d%%" % ('=' * ((i+1) * 50 / num_Rounds_online), (i + 1) * 100 / num_Rounds_online ))
		sys.stdout.flush()
	print("")

	perceptron[:] = perceptron[:]/np.linalg.norm(perceptron[:])
	return training_points

#############
## Parameters:
##	 features : 2D numpy array of features. Every row corresponds to one feature vector
##	 protected_attribute: Numpy array of 0's and 1's as long as number of rows in feature matrix indicating which group the feature belongs to.
##
## Return Type: List of p0, p1. p0 and p1 are fractions of features belonging to groups 0 and 1 labelled as positive by an unbiassed user among
##				all features labelled as positive by the user
##
## This function labels the features using an unbiassed attribute. Among the feature vectors labelled as positive by the user,
## it computes the fraction of feature vectors coming from group 0 (call it p0) and the fraction of feature vectors coming from group 1 (call it p1)
def getBaselineRatios(features, protected_attribute, weights):
	[output_baseline , bias_baseline] = userFeedback(len(features), 0.00, features, weights, protected_attribute)
	smoothing_parameter = len(protected_attribute) * 0.01
	num0 = smoothing_parameter
	num1 = smoothing_parameter
	for i in range(len(protected_attribute)):
		if(output_baseline[i] > 0.5):
			if(protected_attribute[i] > 0.5):
				num1 = num1 + 1
			else:
				num0 = num0 + 1
	p0 = num0 / (num0 + num1)
	p1 = num1 / (num0 + num1)

	return [p0, p1]