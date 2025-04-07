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

from rbf_verify import SVMWrapper

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

def main(ds_name, model_filename, mask_prot_attrs, model_type):
	class_attr, prot_attrs, input_vars, train_x, train_y, test_x, test_y = load_data(ds_name, mask_prot_attrs)

	print('Train Data: X shape = ', train_x.shape, ' Y shape = ', train_y.shape)
	print('Test Data: X shape = ', test_x.shape, ' Y shape = ', test_y.shape)
	
	kernel = 'poly'
	max_iter = 10000

	# Training params
	d = 2 # kernel degree

	if model_type == 1:	
		C, gamma = 1000., 0.0001
	elif model_type == 2:
		C, gamma = 1., 0.5
	
	print('Model type: %d, C = %.5f, gamma = %.5f' % (model_type, C, gamma), flush=True)
	
	if os.path.exists(model_filename) and LOAD_MODEL_IF_EXISTS:
		print('Loading model from file: ', model_filename, flush=True)
		clf = joblib.load(model_filename)
	else:
		print('Training model...', flush=True)
		clf = SVC(kernel = 'rbf', C = C, gamma = gamma)
		print('Time check:', time.strftime("%H:%M:%S", time.localtime()), flush=True)
		clf.fit(train_x, train_y)
		print('Training complete...', flush=True)
		print('Time check:', time.strftime("%H:%M:%S", time.localtime()), flush=True)

		# Save model to file
		joblib.dump(clf, model_filename)
		print('Saved model to file: ', model_filename, flush=True)
	# End if
	
	C     = clf.C
	gamma = clf.gamma

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

	svm2 = SVMWrapper(support_vectors = clf.support_vectors_,
					  multipliers = clf.dual_coef_, bias = clf.intercept_,
					  C = clf.C, gamma = clf.gamma)
	#print('Checks preds: ', np.all(y_pred_train == svm2.predict(train_x)))
	print('Num SV = %d, C = %f, gamma = %f' % (svm2.sv_x.shape[0], svm2.C, svm2.gamma))

	print('\nVERIFYING INDIVIDUAL BIAS\n---------------------------', flush=True)
	print('Time check: ', time.strftime("%H:%M:%S", time.localtime()), flush=True)

	t0 = time.perf_counter()
	
	results = svm2.verify_ind_bias(input_vars)
	
	t1 = time.perf_counter()

	print('Total Time taken: %.3f secs (%.3f mins)' % (t1 - t0, (t1 - t0) / 60.0), flush=True)

	print_results(results, input_vars)

	if 'x' in results:
		Xb_1 = np.expand_dims(results['x'], axis = 0)
		Xb_2 = np.expand_dims(results['y'], axis = 0)
		yb_1, yb_2 = svm2.predict(Xb_1), svm2.predict(Xb_2)

		print('Verifying bias: Pred(x) = %d, Pred(y) = %d' % (yb_1, yb_2))
	# End if
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
	model_type = int(args.setdefault('model-type', '1'))
	
	assert model_type in [1, 2]
	
	main(ds_name, model_filename, mask_prot_attrs, model_type)
# End if
