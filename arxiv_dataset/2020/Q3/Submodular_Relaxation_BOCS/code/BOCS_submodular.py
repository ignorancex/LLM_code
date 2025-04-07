import numpy as np
import cvxpy as cvx
from itertools import combinations
from LinReg import LinReg
from sample_models import sample_models
import time
import itertools
import graph_cuts


def BOCS(inputs, order, AFO, logger):
	# BOCS: Function runs binary optimization using SDP/Semi-definite relaxation on
	# the model drawn from the distribution over beta parameters
	# inputs : dictionary containing atleast following keys
	# 		 : 'n_vars' - dimension of the input space 
	# 		 : 'model' - callable function that returns the value of the black-box objective given an input x 
	# 		 : 'penalty' - L1 penalty for the objective (if any)
	# 		 : 'x_vals' - input points used to initialize the surrogate model 
	# 		 : 'y_vals' - objective values corresponding to x_vals
	# order : order of the Bayesian linear regression model (works with second order only)
	# AFO : type of AFO optimizer, 'submodular-relaxations' or 'SDP-l1'
	# logger : logger object to log data


	logger.debug("Starting of new run")
	# Extract inputs
	n_vars  = inputs['n_vars']
	model   = inputs['model']
	penalty = inputs['penalty']

	# Train initial statistical model
	start_time = time.time()
	LR = LinReg(n_vars, order)
	LR.train(inputs, logger)
	logger.debug("Training time for initial model: %s"%str(time.time()-start_time))

	# Find number of iterations based on total budget
	n_init = inputs['x_vals'].shape[0]
	n_iter = inputs['evalBudget'] - n_init

	# Declare vector to store results
	model_iter = np.zeros((n_iter, n_vars))
	obj_iter   = np.zeros(n_iter)

	for t in range(n_iter):
		logger.debug("Iteration %s"%str(t))
		# Draw alpha vector for Thompson sampling objective
		alpha_t = LR.alpha

		# Run submodular-relaxation based AFO
		if AFO == 'submodular-relaxations':
			start_time = time.time()
			graphobj = graph_cuts.GraphCuts(n_vars, inputs['lambda'], alpha_t, logger)
			x_new, extra_sub, all_points, all_vals = graphobj.get_solution_min_cut(random_set=False, boykov=False, max_t=10)
			x_new = x_new.astype(np.float)
			logger.debug("submodular-relaxations best acq func value - ran for 10 iterations %s"%str(LR.surrogate_model(x_new.reshape(1, n_vars), alpha_t)
																		+penalty(x_new.reshape(1, n_vars))))
			logger.debug("time for submodular-relaxations optimization: %s"%str(time.time()-start_time))

		# Run semidefinite relaxation based AFO
		elif AFO == 'SPD-l1':
			start_time = time.time()
			x_new, _ = sdp_relaxation(alpha_t, inputs)
			logger.debug("time for sdp optimization: %s"%str(time.time()-start_time))
			logger.debug("sdp acq func value %s"%str(LR.surrogate_model(x_new.reshape(1, n_vars), alpha_t)
															+penalty(x_new.reshape(1, n_vars))))	
			logger.debug("sdp selected point")
			logger.debug("%s"%str(x_new))
			logger.debug("function value of sdp selected point: %s"%str(model(x_new.reshape((1,n_vars)))))
		else:
			raise NotImplementedError
		# evaluate model objective at new evaluation point
		x_new = x_new.reshape((1,n_vars))
		y_new = model(x_new)
		print("New point selected: ", x_new)
		print("Objective value at new point: ",y_new)
		#print(model(temp_x_new))
		logger.debug("submodular-relaxations selected point")
		logger.debug("%s"%str(x_new))
		logger.debug("new function value for submodular-relaxations based selected point: %s"%str(y_new))
		# Update inputs dictionary
		inputs['x_vals'] = np.vstack((inputs['x_vals'], x_new))
		inputs['y_vals'] = np.hstack((inputs['y_vals'], y_new))
		inputs['init_cond'] = x_new
		# re-train linear model
		start_time = time.time()
		LR.train(inputs, logger)
		logger.debug("re-training time: %s"%str(start_time-time.time()))
		# Save results for optimal model
		model_iter[t,:] = x_new
		obj_iter[t]	= y_new + penalty(x_new)
		

	return (model_iter, obj_iter)



def sdp_relaxation(alpha, inputs):
	# SDP_Relaxation: Function runs simulated annealing algorithm for 
	# optimizing binary functions. The function returns optimum models and min 
	# objective values found at each iteration

	# Extract n_vars
	n_vars = inputs['n_vars']

	# Extract vector of coefficients
	b = alpha[1:n_vars+1] + inputs['lambda']
	a = alpha[n_vars+1:]

	# get indices for quadratic terms
	idx_prod = np.array(list(combinations(np.arange(n_vars),2)))
	n_idx = idx_prod.shape[0]

	# check number of coefficients
	if a.size != n_idx:
	    raise ValueError('Number of Coefficients does not match indices!')

	# Convert a to matrix form
	A = np.zeros((n_vars,n_vars))
	for i in range(n_idx):
		A[idx_prod[i,0],idx_prod[i,1]] = a[i]/2.
		A[idx_prod[i,1],idx_prod[i,0]] = a[i]/2.

	# Convert to standard form
	bt = b/2. + np.dot(A,np.ones(n_vars))/2.
	bt = bt.reshape((n_vars,1))
	At = np.vstack((np.append(A/4., bt/2.,axis=1),np.append(bt.T,2.)))

	# Run SDP relaxation
	X = cvx.Variable((n_vars+1, n_vars+1), PSD=True)
	obj = cvx.Minimize(cvx.trace(cvx.matmul(At,X)))
	constraints = [cvx.diag(X) == np.ones(n_vars+1)]
	prob = cvx.Problem(obj, constraints)
	prob.solve(solver=cvx.CVXOPT)

	# Extract vectors and compute Cholesky
	# add small identity matrix is X.value is numerically not PSD
	try:
		L = np.linalg.cholesky(X.value)
	except:
		XpI = X.value + 1e-15*np.eye(n_vars+1)
		L = np.linalg.cholesky(XpI)

	# Repeat rounding for different vectors
	n_rand_vector = 100

	model_vect = np.zeros((n_vars,n_rand_vector))
	obj_vect   = np.zeros(n_rand_vector)

	for kk in range(n_rand_vector):

		# Generate a random cutting plane vector (uniformly 
		# distributed on the unit sphere - normalized vector)
		r = np.random.randn(n_vars+1)
		r = r/np.linalg.norm(r)
		y_soln = np.sign(np.dot(L.T,r))

		# convert solution to original domain and assign to output vector
		model_vect[:,kk] = (y_soln[:n_vars]+1.)/2.
		obj_vect[kk] = np.dot(np.dot(model_vect[:,kk].T,A),model_vect[:,kk]) \
			+ np.dot(b,model_vect[:,kk])
	# Find optimal rounded solution
	opt_idx = np.argmin(obj_vect)
	model = model_vect[:,opt_idx]
	obj   = obj_vect[opt_idx]

	return (model, obj)


# -- END OF FILE --
