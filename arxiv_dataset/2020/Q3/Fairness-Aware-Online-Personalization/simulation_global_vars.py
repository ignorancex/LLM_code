## This file constains all the global variables required for running the simulations.
## Variable specific for any experiment are defined within functions

import numpy as np

# Number of rounds to train perceptron online
num_Rounds_online = 1000

# Number of rounds to train perceptron for warm start
num_Rounds_warm = 1000

# Weights chosen by the user to various attributes and constant term
weights = np.array([0.35, 0.28, 0.28, 0.48])

# Number of data points
num_data = 12000

# Probability of data points belonging to group 1
p_groups = 0.50

# Mean and Variance for generating the attributes 1,2...m (Attribute 0 is from uniform distribution) for the 2 protected groups
mu = [[0.35, 0.65],[0.65, 0.35]]
sigma = [[0.12, 0.12],[0.12, 0.12]]

# Number of runs of simulation to average over before generating plots
RUNS = 10

#############
## Parameters for the evolution experiement
def parameters_for_evolution_exp():
	# learning rates to experiment over for a fixed bias
	learning_rates = [0.01, 0.02, 0.03, 0.04 ]

	# bias of user to experiment over for a fixed learning rate
	exp_bias = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]

	# list of rounds after which we want to check the metrics
	probe = [25 * i for i in range(1, 40)]

	return [learning_rates, exp_bias, probe]

#############
## Parameters for the fairness and precision experiement
def parameters_for_fairness_and_precision_after_online_training_exp():
	# learning rates to experiment over to see how fair the model gets
	learning_rates = [i * 0.001 for i in range(1,40,5)] + [i * 0.01 for i in range(4,10,1)]
	####learning_rates = [0.05, 0.10]

	# bias of user to experiment
	exp_bias = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]

	# list of k-values to compute skew@k, precision@k
	probe = [25, 50, 100, 500, 1000]

	return [learning_rates, exp_bias, probe]

#############
## Parameters for the regularised fairness and precision experiement
def parameters_for_fairness_and_precision_after_regularised_online_training_exp():
	# regularisation parameter to experiment over to see how fair the model gets
	reg_lambda = [i for i in range(0, 100, 10)]

	# bias of user to experiment
	exp_bias = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]

	# list of k-values to compute skew@k, precision@k
	probe = [25, 50, 100, 500, 1000]

	return [reg_lambda, exp_bias, probe]

