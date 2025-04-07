#!/usr/bin/env python3

from joblib import dump, load
import pickle
import numpy as np

model_config = pickle.load(open("./config.pickle", "rb"))

clf = load('../acc_predictor.joblib') 

acc = []
for cfg in model_config:
	res = [cfg['resolution']]
	for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
		res += cfg[k]
	#print(clf.predict(np.asarray(res).reshape((1, -1))))
	acc.append(clf.predict(np.asarray(res).reshape((1, -1)))[0])
print(acc)
pickle.dump(acc, open("accuracy.pickle", "wb"))