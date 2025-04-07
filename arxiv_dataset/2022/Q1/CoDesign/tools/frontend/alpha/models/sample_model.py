#!/usr/bin/env python3

import random
import pickle

resolution = [192, 224, 256, 288]
width = [16, 16, 24, 32, 64, 112, 192, 216, 1792]
kernel_size = [3, 3, 3, 3, 3, 3, 3]
expand_ratio = [1, 4, 4, 4, 4, 6, 6]
depth = [1, 3, 3, 3, 3, 3, 1]



n_model = 10000
config = []
for _ in range(n_model):
	tmp = {}
	random.seed(_)
	r = random.choice(resolution)
	#print(r)
	k = [random.choice([3, 5, 7]) for i in range(5)]
	k = [3] + k + [3]
	#print(k)
	e = [random.choice([4, 5, 6]) for i in range(5)]
	e = [1] + e + [6]
	#print(e)
	d = [random.choice([3, 4, 6]) for i in range(5)]
	d = [1] + d + [1]
	#print(d)
	#print(h)
	tmp["resolution"] = r
	tmp["width"] = width
	tmp["kernel_size"] = k
	tmp["expand_ratio"] = e
	tmp["depth"] = d
	if tmp not in config:
		config.append(tmp)
	
print(config)
pickle.dump(config, open("./config.pickle", "wb"))