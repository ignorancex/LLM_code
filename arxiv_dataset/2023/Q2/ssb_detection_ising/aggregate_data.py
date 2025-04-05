import os
import json
import time
import numpy as np
from phasefinder import compressor

data_dir = "data"
Js = ["ferromagnetic", "antiferromagnetic"]
Ls = [16, 32, 64, 128]

for J in Js:
	print("{}:".format(J))
	for L in Ls:
		print("L = {:d}".format(L))
		time_0 = time.time()

		states = []
		times = {}
		L_dir = os.path.join(data_dir, J, "L{:d}".format(L))
		output_dir = os.path.join(L_dir, "aggregate")
		os.makedirs(output_dir, exist_ok=True)

		for dir in sorted(os.listdir(L_dir)):
			if dir[0] != "T":
				continue
			with open(os.path.join(L_dir, dir, "states.enc"), "r") as fp:
				X = fp.read()
			X = compressor.decode(X)
			X = np.array([char for char in X]).reshape((-1, L**2))
			states.append(X)

		states = np.stack(states, 0)
		states = 2*states.astype(np.float32)-1
		time_1 = time.time()
		np.savez(os.path.join(output_dir, "states.npz"), train=states[:,::-2], test=states[:,-2::-2])
		times["states"] = time.time()-time_0
		time_2 = time.time()

		states = states.reshape((states.shape[0], states.shape[1], L, L))
		states_symmetric = 2/L**2 * np.stack([ \
			np.sum(states[:,:,::2,::2]+states[:,:,1::2,1::2], (2, 3)), \
			np.sum(states[:,:,1::2,::2]+states[:,:,::2,1::2], (2, 3)) \
		], 2)
		np.savez(os.path.join(output_dir, "states_symmetric.npz"), train=states_symmetric[:,::-2], test=states_symmetric[:,-2::-2])
		times["states_symmetric"] = time.time()-time_0 - (time_2-time_1)

		with open(os.path.join(output_dir, "times.json"), "w") as fp:
			json.dump(times, fp, indent=2)

print("All done!")
