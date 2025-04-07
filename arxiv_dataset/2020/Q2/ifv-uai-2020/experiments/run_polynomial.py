# Author: Philips George John

import sys, os

sys.path.insert(0, '../dev-pkgs')
sys.path.insert(0, '../research')
sys.path.insert(0, '.')

import time
import shlex

import numpy as np
import pandas as pd
import random as rnd

import joblib
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix

from poly_verify import svm2poly, verify

from ibv.model import InputVariable
from helpers import print_results

from data_loader import load_data


# Ensure reproducibility of results
np.random.seed(42) # numpy RNG
rnd.seed(12345)    # core python RNG

###############################
########	MAIN 	###########
###############################

LOAD_MODEL_IF_EXISTS = True

def main(ds_name, model_filename, mask_prot_attrs):
	class_attr, prot_attrs, input_vars, train_x, train_y, test_x, test_y = load_data(ds_name, mask_prot_attrs)

	print('Train Data: X shape = ', train_x.shape, ' Y shape = ', train_y.shape)
	print('Test Data: X shape = ', test_x.shape, ' Y shape = ', test_y.shape)
	
	kernel = 'poly'
	max_iter = 10000

	# Training params
	d = 2 # kernel degree
	C, gamma, r = 1., 0.001, 0.
	
	if os.path.exists(model_filename) and LOAD_MODEL_IF_EXISTS:
		print('Loading model from file: ', model_filename, flush=True)
		clf = joblib.load(model_filename)
	else:
		print('Training model...', flush=True)
		clf = SVC(kernel=kernel, verbose=True, max_iter=max_iter,
				  C=C, gamma=gamma, coef0=r, degree=d)
		print('Time check:', time.strftime("%H:%M:%S", time.localtime()), flush=True)
		clf.fit(train_x, train_y)
		print('Training complete...', flush=True)
		print('Time check:', time.strftime("%H:%M:%S", time.localtime()), flush=True)

		# Save model to file
		joblib.dump(clf, model_filename)
		print('Saved model to file: ', model_filename, flush=True)
	# End if
	
	d     = clf.degree
	C     = clf.C
	gamma = clf.gamma
	r     = clf.coef0

	y_pred_train = clf.predict(train_x)
	y_pred_test  = clf.predict(test_x)

	acc = accuracy_score(train_y, y_pred_train)
	cm = confusion_matrix(train_y, y_pred_train)

	print('Confusion Matrix (Train):', cm)
	print('Accuracy score (Train):', acc)

	acc = accuracy_score(test_y, y_pred_test)
	cm = confusion_matrix(test_y, y_pred_test)

	print('Confusion Matrix (Test):', cm)
	print('Accuracy score (Test):', acc)

	print('\nBuilding polynomial model...', flush=True)
	svm_P = svm2poly(sv_weights = np.ravel(clf.dual_coef_), sv_x = clf.support_vectors_, bias = clf.intercept_, gamma = gamma, r = r, d = d)
	
	# svm_P_preds = np.zeros_like(y_pred_train)
	# for i in range(train_x.shape[0]):
	# 	y_i = svm_P.evaluate(train_x[i])
	# 	svm_P_preds[i] = (y_i > 0).astype(np.int32)
	# print('Checks preds: ', np.all(y_pred_train == svm_P_preds), flush = True)
	
	print('Num SV = %d, C = %f, gamma = %f, r = %f, d = %d' % (clf.support_vectors_.shape[0], C, gamma, r, d), flush=True)

	print('\nVERIFYING INDIVIDUAL BIAS\n---------------------------', flush=True)
	print('Time check: ', time.strftime("%H:%M:%S", time.localtime()), flush=True)
	t0 = time.perf_counter()

	# Do bias verification here
	res = verify(svm_P, input_vars)

	t1 = time.perf_counter()
	print('Total Time taken: %.3f secs (%.3f mins)' % (t1 - t0, (t1 - t0) / 60.0), flush=True)

	no_bias = True
	for r in res:
		if r != 0:
			print('FINAL RESULT: Possible bias.', flush=True)
			no_bias = False
	# End for
	if no_bias:
		print('FINAL RESULT: No bias.', flush=True)
	
	print('END', flush=True)
# End fn main

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print('Error: Usage: python %s <ds-name> <model-filename> [mask-prot-attrs=true]' % sys.argv[0])
	# End if
	
	ds_name = sys.argv[1].lower()
	model_filename = sys.argv[2]
	
	args = {}
	if len(sys.argv) > 3:
		opt_arg_str = ' '.join(sys.argv[3:])
		args = dict(token.split('=', 1) for token in shlex.split(opt_arg_str))
	# End if
	
	mask_prot_attrs = (args.setdefault('mask-prot-attrs', 'false').lower() == 'true')
	
	main(ds_name, model_filename, mask_prot_attrs)
# End if
