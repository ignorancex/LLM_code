import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from model import *
from metrics import *
from online_perceptron import *
from simulation_global_vars import *
import sys

matplotlib.rcParams.update({'figure.figsize': (20,10), 'font.size': 25, 'legend.fontsize': 15, 'legend.shadow': False, 'legend.framealpha': 0.5, 'xtick.labelsize': 25, 'ytick.labelsize': 25})

#############
## Parameters:
##	 features : 2D numpy array of features. Every row corresponds to one feature vector
##	 protected_attribute: Numpy array of 0's and 1's as long as number of rows in feature matrix indicating which group the feature belongs to.
##	 output: 
##   pw: warm-start weights vector of perceptron
##	 percep: online weights vector of perceptron
##
## Return Type: List of skew_warm, ndcs_warm, skew_ol, ndcs_ol, precision_warm, precision_ol.
##				skew_warm is numpy array of skew values @ probe for warm start model
##				ndcs_warm is ndcs for warm start model
##				skew_ol is numpy array of skew values @ probe for online learnt model
##				ndcs_ol is ndcs for online learnt model
##				precision_warm is numpy array of precision values @ probe for warm start model
##				precision_ol is numpy array of precision values @ probe for online learnt model
##
## This function computes the fairness and precision metrics for the warm start model and online learnt model.
def getSkewPrecision(features, protected_attribute, output, pw, percep):
	## WARM START FAIRNESS
	visited = np.zeros(num_data)
	top_indices_warm = get_top_k(np.max(probe), features, pw, visited)

	skew_warm = np.zeros(len(probe))
	for i in range(len(probe)):
		skew_warm[i] = skew(probe[i], top_indices_warm, protected_attribute, p0, p1)
	ndcs_warm = NDCS(top_indices_warm, protected_attribute, p0, p1)

	## WARM START PRECISION
	precision_warm = np.zeros(len(probe))
	for i in range(len(probe)):
		precision_warm[i] = precision(probe[i], top_indices_warm, output)

	visited = np.zeros(num_data)
	top_indices_ol = get_top_k(np.max(probe), features, percep, visited)

	## POST-ONLINE TRAINING FAIRNESS
	skew_ol = np.zeros(len(probe))
	for i in range(len(probe)):
		skew_ol[i] = skew(probe[i], top_indices_ol, protected_attribute, p0, p1)

	ndcs_ol = NDCS(top_indices_ol, protected_attribute, p0, p1)

	## POST-ONLINE TRAINING PRECISION
	precision_ol = np.zeros(len(probe))
	for i in range(len(probe)):
		precision_ol[i] = precision(probe[i], top_indices_ol, output)

	return [skew_warm, ndcs_warm, skew_ol, ndcs_ol, precision_warm, precision_ol]

#############
## Parameters:
##	 features : 2D numpy array of features. Every row corresponds to one feature vector
##	 protected_attribute: Numpy array of 0's and 1's as long as number of rows in feature matrix indicating which group the feature belongs to.
##	 p_bias: probability of user being biassed. When user is biassed, his decision is based on which group the feature belongs to.
##   pw: warm-start weights vector of perceptron
##	 lr: learning rate
##	 probe: array of values at which skew and precision have to be computed (For Skew@k, Precision@k......List of k values)
##
## Return Type: List of skew_warm, ndcs_warm, skew_ol, ndcs_ol, precision_warm, precision_ol.
##				skew_warm is numpy array of skew values @ probe for warm start model
##				ndcs_warm is ndcs for warm start model
##				skew_ol is numpy array of skew values @ probe for online learnt model
##				ndcs_ol is ndcs for online learnt model
##				precision_warm is numpy array of precision values @ probe for warm start model
##				precision_ol is numpy array of precision values @ probe for online learnt model
##
## This function runs the fairness experiment. The user model defined in model.py is used to
## label the feature vectors. The warm start perceptron is trained online by calling the function onlineTrainingProcess
## After online learning fairness and precision metrics are computed for the warm start model and online learnt model.
def exp_fairness_and_precision_after_online_training(features, protected_attribute, p_bias, pw, wt_protected, lr, probe):
	print("--------------------------------------")
	[output , bias] = userFeedback(num_data, p_bias, features, weights, protected_attribute)
	visited = np.zeros(num_data)

	print("User Bias: "+ "%.2f" %(100 * p_bias)+"%")

	print("Perceptron(Warm Start):")
	print(pw)

	percep = np.zeros_like(pw)
	percep[:] = pw
	onlineTrainingProcess(features, output, percep, lr, visited, num_Rounds_online)

	print("Perceptron(Warm Start):")
	print(pw)

	print("Perceptron(Warm Start + Online Learning):")
	print(percep)

	[skew_warm, ndcs_warm, skew_ol, ndcs_ol, precision_warm, precision_ol] = getSkewPrecision(features, protected_attribute, output, pw, percep)
	return [skew_warm, ndcs_warm, skew_ol, ndcs_ol, precision_warm, precision_ol]

#############
## Parameters:
##	 features : 2D numpy array of features. Every row corresponds to one feature vector
##	 protected_attribute: Numpy array of 0's and 1's as long as number of rows in feature matrix indicating which group the feature belongs to.
##	 p_bias: probability of user being biassed. When user is biassed, his decision is based on which group the feature belongs to.
##   pw: warm-start weights vector of perceptron
##   wt_protected: Weights of a linear model which tries to estimate protected_attribute from feature vectors
##   reg: Regularization Parameter (lambda) for fairness
##	 probe: array of values at which skew and precision have to be computed (For Skew@k, Precision@k......List of k values)
##
## Return Type: List of skew_warm, ndcs_warm, skew_ol, ndcs_ol, precision_warm, precision_ol.
##				skew_warm is numpy array of skew values @ probe for warm start model
##				ndcs_warm is ndcs for warm start model
##				skew_ol is numpy array of skew values @ probe for online learnt model
##				ndcs_ol is ndcs for online learnt model
##				precision_warm is numpy array of precision values @ probe for warm start model
##				precision_ol is numpy array of precision values @ probe for online learnt model
##
## This function runs the regularised fairness experiment. The user model defined in model.py is used to
## label the feature vectors. The warm start perceptron is trained online by calling the function onlineFairTrainingProcess
## After regularised online learning fairness and precision metrics are computed for the warm start model and online learnt model.
def exp_fairness_and_precision_after_regularised_online_training(features, protected_attribute, p_bias, pw, wt_protected, reg, probe):
	lr = 0.04
	print("--------------------------------------")
	[output , bias] = userFeedback(num_data, p_bias, features, weights, protected_attribute)
	n = -1 * np.ones(len(features))
	n = n.reshape(-1, 1)
	extended_features = np.append(features, n, axis=1)
	[sigma, sigma0, sigma1] = getCovariance(extended_features, output)

	visited = np.zeros(num_data)

	print("User Bias: "+ "%.2f" %(100 * p_bias)+"%")

	print("Perceptron(Warm Start):")
	print(pw)

	percep = np.zeros_like(pw)
	percep[:] = pw
	onlineTrainingProcessFair(features, output, percep, lr, visited, num_Rounds_online, sigma, reg, wt_protected)

	print("Perceptron(Warm Start):")
	print(pw)

	print("Perceptron(Warm Start + Online Learning):")
	print(percep)

	[skew_warm, ndcs_warm, skew_ol, ndcs_ol, precision_warm, precision_ol] = getSkewPrecision(features, protected_attribute, output, pw, percep)
	return [skew_warm, ndcs_warm, skew_ol, ndcs_ol, precision_warm, precision_ol]

#############
## Parameters:
##	 features : 2D numpy array of features. Every row corresponds to one feature vector
##	 protected_attribute: Numpy array of 0's and 1's as long as number of rows in feature matrix indicating which group the feature belongs to.
##	 p_bias: probability of user being biassed. When user is biassed, his decision is based on which group the feature belongs to.
##   pw: warm-start weights vector of perceptron
##	 lr: learning rate
##	 probe: list of number of rounds after which we check how fairness evolves
##
## Return Type: List of skew values after t rounds for t in rounds-parameter
##
## This function checks how fairness evolves with rounds. The user model defined in model.py is used to
## label the feature vectors. The warm start perceptron is trained online by calling the function onlineTrainingProcess
## After every few rounds (defined in the probe parameter), all the training points used till that time are collected
## and skew is computed over them. These skew values are returned in a list
def exp_how_fairness_evolves(features, protected_attribute, p_bias, pw, lr, probe):
	print("--------------------------------------")
	[output , bias] = userFeedback(num_data, p_bias, features, weights, protected_attribute)

	visited = np.zeros(num_data)

	print("User Bias: "+ "%.2f" %(100 * p_bias)+"%")

	print("Perceptron(Warm Start):")
	print(pw)

	percep = np.zeros_like(pw)
	percep[:] = pw

	training_points = []
	skew_val = []
	for i in range(len(probe)):
		if(i==0):
			training_points = training_points + onlineTrainingProcess(features, output, percep, lr, visited, probe[i])
		else:
			training_points = training_points + onlineTrainingProcess(features, output, percep, lr, visited, probe[i] - probe[i-1])
		s = skew(probe[i], training_points, protected_attribute, p0, p1)
		skew_val = skew_val + [s]
	return skew_val


#############EXPERIMENT-1##############################################
[learning_rates, exp_bias, probe] = parameters_for_evolution_exp()

evolution_results_avg = np.zeros((len(learning_rates), len(exp_bias), len(probe)))

for runs in range(RUNS):
	[features0, protected_attribute0] = data_generation(num_data, p_groups, mu, sigma)
	[features1, protected_attribute1] = data_generation(num_data, p_groups, mu, sigma)

	[output0 , bias0] = userFeedback(num_data, 0.00, features0, weights, protected_attribute0)

	[p0, p1] = getBaselineRatios(features1, protected_attribute1, weights)

	evolution_results = np.zeros((len(learning_rates), len(exp_bias), len(probe)))

	for j in range(len(learning_rates)):
		for i in range(len(exp_bias)):
			lr = learning_rates[j]
			perceptron = np.random.uniform(0, 1, 4)
			perceptron = constructWarmStartModelL2Reg(features0, output0, 0.00)
			perceptron_warm = np.zeros_like(perceptron)
			perceptron_warm[:] = perceptron

			evolution_results[j][i][:] = exp_how_fairness_evolves(features1, protected_attribute1, exp_bias[i], perceptron_warm, lr, probe)

	evolution_results_avg = evolution_results_avg + evolution_results

evolution_results_avg = evolution_results_avg / RUNS

plt.figure()
for i in range(len(exp_bias)):
	plt.plot(probe, evolution_results_avg[0,i,:], label = "warm_start with bias "+str(exp_bias[i]), linewidth = 2, marker='o', linestyle='-')
plt.title("Skew Vs Training Rounds: learning rate of "+str(learning_rates[0]))
plt.xlabel("Training Rounds")
plt.ylabel("Skew")
plt.xlim([0.0, 1000.0])
plt.ylim([-0.5, 1.0])
plt.legend(loc="lower right")
plt.savefig("Evolution(fixed learning rate).pdf")
plt.close()

plt.figure()
for i in range(len(learning_rates)):
	plt.plot(probe, evolution_results_avg[i,2,:], label = "learning_rate "+str(learning_rates[i]), linewidth = 2, marker='o', linestyle='-')
plt.title("Skew Vs Training Rounds: warm_start with bias "+str(exp_bias[2]))
plt.xlabel("Training Rounds")
plt.ylabel("Skew")
plt.xlim([0.0, 1000.0])
plt.ylim([-0.5, 1.0])
plt.legend(loc="lower right")
plt.savefig("Evolution(fixed bias).pdf")
plt.close()

#############EXPERIMENT-2##############################################
[learning_rates, exp_bias, probe] = parameters_for_fairness_and_precision_after_online_training_exp()

fairness_results_avg = np.zeros((len(learning_rates), 1 + len(exp_bias), 1 + len(probe)))
precision_results_avg = np.zeros((len(learning_rates), 2 * len(exp_bias), len(probe)))

for runs in range(RUNS):
	### GENERATE DATA
	[features0, protected_attribute0] = data_generation(num_data, p_groups, mu, sigma)
	[features1, protected_attribute1] = data_generation(num_data, p_groups, mu, sigma)

	### LABEL DATA FOR WARM START
	[output0 , bias0] = userFeedback(num_data, 0.00, features0, weights, protected_attribute0)

	### BASELINE CALCULATIONS
	[p0, p1] = getBaselineRatios(features1, protected_attribute1, weights)
	print("Protected Grp 0: " + "%.4f" %(p0))
	print("Protected Grp 1: " + "%.4f" %(p1))

	fairness_results = np.zeros((len(learning_rates), 1 + len(exp_bias), 6))
	precision_results = np.zeros((len(learning_rates), 2 * len(exp_bias), 5))

	for i in range(len(learning_rates)):
		lr = learning_rates[i]
		perceptron = np.random.uniform(0, 1, 4)

		perceptron = constructWarmStartModelL2Reg(features0, output0, 0.00)
		perceptron_warm = np.zeros_like(perceptron)
		perceptron_warm[:] = perceptron

		w = np.random.uniform(0, 1, 4)

		wt_protected = constructWarmStartModelL2Reg(features0, protected_attribute0, 0.02)
		wt_protected[:] = wt_protected[:]/np.linalg.norm(wt_protected[:])

		print("Learning Rate:")
		print(lr)

		print("Protected Weights:")
		print(wt_protected)

		print("Weights:")
		print(weights)

		for j in range(len(exp_bias)):
			tmp = exp_fairness_and_precision_after_online_training(features1, protected_attribute1, exp_bias[j], perceptron_warm, wt_protected, lr, probe)
			fairness_results[i][0][:-1] = tmp[0]
			fairness_results[i][0][-1] = tmp[1]
			fairness_results[i][j+1][:-1] = tmp[2]
			fairness_results[i][j+1][-1] = tmp[3]

			precision_results[i][2 * j][:] = tmp[4]
			precision_results[i][2 * j+1][:] = tmp[5]

	fairness_results_avg = fairness_results_avg + fairness_results
	precision_results_avg = precision_results_avg + precision_results

fairness_results_avg = fairness_results_avg / RUNS
precision_results_avg = precision_results_avg / RUNS

metrics = ['Skew@25', 'Skew@50', 'Skew@100', 'Skew@500', 'Skew@1000', 'NDCS']
for j in range(len(metrics)):
	plt.figure()
	for i in range(len(exp_bias)+1):
		if(i==0):
			plt.plot(learning_rates, fairness_results_avg[:,i,j], label = "warm_start", linewidth = 2, marker='o')
		else:
			plt.plot(learning_rates, fairness_results_avg[:,i,j], label = "warm + online with bias "+str(exp_bias[i-1]), linewidth = 2, marker='o')
	plt.title(metrics[j]+" Vs Learning Rate")
	plt.xlabel("Learning Rate")
	plt.ylabel(metrics[j])
	plt.xlim([0.0, 0.1])
	plt.ylim([-0.5, 1.0])
	plt.legend(loc="lower right")
	plt.savefig(metrics[j]+".pdf")
	plt.close()

c = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
metrics = ['Precision@25', 'Precision@50', 'Precision@100', 'Precision@500', 'Precision@1000']
for j in range(len(metrics)):
	plt.figure()
	for i in range(len(exp_bias)):
		plt.plot(learning_rates, precision_results_avg[:,2 * i,j], label = "warm_start with bias "+str(exp_bias[i]), linewidth = 2, marker='o', linestyle='--', color=c[i])
		plt.plot(learning_rates, precision_results_avg[:,2 * i + 1,j], label = "warm + online with bias "+str(exp_bias[i]), linewidth = 2, marker='o', linestyle='-', color=c[i])
	plt.title(metrics[j]+" Vs Learning Rate")
	plt.xlabel("Learning Rate")
	plt.ylabel(metrics[j])
	plt.xlim([0.0, 0.1])
	plt.ylim([0.0, 1.2])
	plt.legend(loc="lower right")
	plt.savefig(metrics[j]+".pdf")
	plt.close()


#############EXPERIMENT-3##############################################
[reg_lambda, exp_bias, probe] = parameters_for_fairness_and_precision_after_regularised_online_training_exp()

fairness_results_avg = np.zeros((len(reg_lambda), 1 + len(exp_bias), 1 + len(probe)))
precision_results_avg = np.zeros((len(reg_lambda), 2 * len(exp_bias), len(probe)))

for runs in range(RUNS):
	### GENERATE DATA
	[features0, protected_attribute0] = data_generation(num_data, p_groups, mu, sigma)
	[features1, protected_attribute1] = data_generation(num_data, p_groups, mu, sigma)

	### LABEL DATA FOR WARM START
	[output0 , bias0] = userFeedback(num_data, 0.00, features0, weights, protected_attribute0)

	### BASELINE CALCULATIONS
	[p0, p1] = getBaselineRatios(features1, protected_attribute1, weights)
	print("Protected Grp 0: " + "%.4f" %(p0))
	print("Protected Grp 1: " + "%.4f" %(p1))

	fairness_results = np.zeros((len(reg_lambda), 1 + len(exp_bias), 6))
	precision_results = np.zeros((len(reg_lambda), 2 * len(exp_bias), 5))

	for i in range(len(reg_lambda)):
		reg = reg_lambda[i]
		perceptron = np.random.uniform(0, 1, 4)

		perceptron = constructWarmStartModelL2Reg(features0, output0, 0.00)
		perceptron_warm = np.zeros_like(perceptron)
		perceptron_warm[:] = perceptron

		w = np.random.uniform(0, 1, 4)

		wt_protected = constructWarmStartModelL2Reg(features0, protected_attribute0, 0.02)
		wt_protected[:] = wt_protected[:]/np.linalg.norm(wt_protected[:])

		print("Lambda:")
		print(reg)

		print("Protected Weights:")
		print(wt_protected)

		print("Weights:")
		print(weights)

		for j in range(len(exp_bias)):
			tmp = exp_fairness_and_precision_after_regularised_online_training(features1, protected_attribute1, exp_bias[j], perceptron_warm, wt_protected, reg, probe)
			fairness_results[i][0][:-1] = tmp[0]
			fairness_results[i][0][-1] = tmp[1]
			fairness_results[i][j+1][:-1] = tmp[2]
			fairness_results[i][j+1][-1] = tmp[3]

			precision_results[i][2 * j][:] = tmp[4]
			precision_results[i][2 * j+1][:] = tmp[5]

	fairness_results_avg = fairness_results_avg + fairness_results
	precision_results_avg = precision_results_avg + precision_results

fairness_results_avg = fairness_results_avg / RUNS
precision_results_avg = precision_results_avg / RUNS


metrics = ['Skew@25', 'Skew@50', 'Skew@100', 'Skew@500', 'Skew@1000', 'NDCS']
for j in range(len(metrics)):
	plt.figure()
	for i in range(len(exp_bias)+1):
		if(i==0):
			plt.plot(reg_lambda, fairness_results_avg[:,i,j], label = "warm_start", linewidth = 2, marker='o')
		else:
			plt.plot(reg_lambda, fairness_results_avg[:,i,j], label = "warm + online with bias "+str(exp_bias[i-1]), linewidth = 2, marker='o')
	plt.title(metrics[j]+" Vs Lambda")
	plt.xlabel("Lambda")
	plt.ylabel(metrics[j])
	plt.xlim([0, 100])
	plt.ylim([-0.5, 1.0])
	plt.legend(loc="lower left")
	plt.savefig("Reg_"+metrics[j]+".pdf")
	plt.close()

c = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
metrics = ['Precision@25', 'Precision@50', 'Precision@100', 'Precision@500', 'Precision@1000']
for j in range(len(metrics)):
	plt.figure()
	for i in range(len(exp_bias)):
		plt.plot(reg_lambda, precision_results_avg[:,2 * i,j], label = "warm_start with bias "+str(exp_bias[i]), linewidth = 2, marker='o', linestyle='--', color=c[i])
		plt.plot(reg_lambda, precision_results_avg[:,2 * i + 1,j], label = "warm + online with bias "+str(exp_bias[i]), linewidth = 2, marker='o', linestyle='-', color=c[i])
	plt.title(metrics[j]+" Vs Lambda")
	plt.xlabel("Lambda")
	plt.ylabel(metrics[j])
	plt.xlim([0, 100])
	plt.ylim([0.0, 1.2])
	plt.legend(loc="lower left")
	plt.savefig("Reg_"+metrics[j]+".pdf")
	plt.close()

