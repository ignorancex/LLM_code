# Author: Philips George John

import numpy as np
import sklearn.metrics as metrics
import numbers

def print_results(inputs, input_vars):
	if isinstance(inputs, list):
		if len(inputs) == 0:
			print('No bias instances found.')
			return
		else:
			for input in inputs:
				_print_instance(input, input_vars)
	else:
		_print_instance(inputs, input_vars)
# End print_results

def _print_instance(input, input_vars):
	if 'x' not in input or 'y' not in input:
		print('No bias instance found.', flush=True)
		return
	
	print(' %-32s  %-16s  %-16s' % ('        Attribute Name', '        x', '        y'))
	print(' %s  %s  %s' % ('-'*32, '-'*16, '-'*16))
	for i, var in enumerate(input_vars):
		if var.type == 'integer':
			print(' %-32s  %16d  %16d' % (var.name[:32], input['x'][i], input['y'][i]))
		elif var.type == 'real':
			print(' %-32s  %16.3f  %16.3f' % (var.name[:32], input['x'][i], input['y'][i]))
	print('', flush = True)
# End _print_instance


def list_ndarray2nested_list(a):
	if isinstance(a, (list,)):
		a_new = []
		for a_item in a:
			l = list_ndarray2nested_list(a_item)
			a_new.append(l)
		return a_new
	elif isinstance(a, np.ndarray):
		return a.tolist()
	elif isinstance(a, np.generic):
		return np.asscalar(a)
	else:
		return a
# End list_ndarray2nested_list

def evaluate_classifier(y_true, y_pred, is_binary = False, indent = 1):
	spacer = ' ' * (indent * 4)
	if y_pred.ndim > 2: return
	
	if y_pred.ndim == 2 and y_pred.shape[1] > 1: # Predictions are probabilities
		y_pred = np.argmax(y_pred, axis = 1)
	
	if y_true.ndim == 2 and y_true.shape[1] > 1: # Labels are one-hot
		y_true = np.argmax(y_true, axis = 1)
	
	average = 'binary' if is_binary else 'weighted'
	
	print('%s%-32s : %.3f' % (spacer, 'Accuracy', metrics.accuracy_score(y_true, y_pred)))
	print('%s%-32s : %.3f' % (spacer, 'F1 Score ({})'.format(average),
			metrics.f1_score(y_true, y_pred, average = average)))
	
	if is_binary:
		print('%s%-32s : %.3f' % (spacer, 'Precision', metrics.precision_score(y_true, y_pred, average = average)))
		print('%s%-32s : %.3f' % (spacer, 'Recall', metrics.recall_score(y_true, y_pred, average = average)))
	print('\n')
	CM = metrics.confusion_matrix(y_true, y_pred)
	print(spacer + 'Confusion Matrix')
	print_matrix(CM, indent + 1)
# End evaluate_classifier
