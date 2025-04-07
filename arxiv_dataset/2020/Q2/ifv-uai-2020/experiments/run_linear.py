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
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

from lin_verify import verify

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
	
	if os.path.exists(model_filename) and LOAD_MODEL_IF_EXISTS:
		print('Loading model from file: ', model_filename)
		clf = joblib.load(model_filename)
	else:
		print('Training model...')
		clf = LogisticRegression()
		clf.fit(train_x, train_y)

		# Save model to file
		joblib.dump(clf, model_filename)
	# End if

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

	w = np.ravel(clf.coef_)
	b = clf.intercept_[0]

	print('VERIFYING INDIVIDUAL BIAS\n--------------------------')
	print('Time check: ', time.strftime("%H:%M:%S", time.localtime()))

	t0 = time.perf_counter()

	res = verify(w, b, input_vars)

	t1 = time.perf_counter()

	print('Time taken: %.3f secs (%.3f mins)' % (t1 - t0, (t1 - t0) / 60.0))

	print_results(res[1], input_vars)
	
	if 'x' in res[1]:
		Xb_1 = np.expand_dims(res[1]['x'], axis = 0)
		Xb_2 = np.expand_dims(res[1]['y'], axis = 0)
		yb_1, yb_2 = clf.predict(Xb_1), clf.predict(Xb_2)

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
	
	main(ds_name, model_filename, mask_prot_attrs)
# End if
