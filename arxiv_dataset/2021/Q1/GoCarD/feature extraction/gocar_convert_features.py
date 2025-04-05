# Convert the dictionary saved by pickle into a dictionary, 
# with constant size arrays as values

import pandas as pd
import pickle
import numpy as np
import glob

from gocar_class_mapping import get_class, get_index

num_classes = 28


def load_file(path):
	with open(path, 'rb') as handle:
		return pickle.load(handle)

def convert(dictionary, num_objects=10):
	# dictionary is k:v pairs. keys are the sampled frame in ms
	# value is a list of all detected objects.
	def one_hot(index, num_classes):
		zeros = np.zeros(num_classes)
		zeros[index] = 1
		return zeros
	def preprocess_tuple(tuple):
		class_str, confidence, t1, t2, t3 = tuple
		one_hot_class = one_hot(get_index(class_str), num_classes)
		rest = np.array([confidence, 
			t1[0],t1[1],t2[0], t2[1], t3[0], t3[1]])
		out = np.concatenate((one_hot_class, rest))
		return out.flatten()

	def map_value(objects):
		objects.sort(key=lambda x: x[1]) # sort according to confidence
		objects = objects[:num_objects]
		objects = [preprocess_tuple(t) for t in objects]
		objects = np.array(objects)
		objects = objects.flatten()
		a = np.zeros(num_objects*(num_classes+7))
		a[0:objects.shape[0]] = objects
		return a


	return {k: map_value(v) for k,v in dictionary.items()}

def save_dict(dictionary, path):
	with open(path, 'wb') as handle:
		pickle.dump(dictionary, handle, protocol = pickle.HIGHEST_PROTOCOL)

def get_all_files(path):
	return glob.glob('{}/*.pickle'.format(path))
	

def convert_gocar(src_path, dst_path):
	all_files = get_all_files(src_path)

	for f in all_files:
		d = convert(load_file(f))
		f = f.split('/')[-1]
		save_dict(d, "{}/{}".format(dst_path, f))

