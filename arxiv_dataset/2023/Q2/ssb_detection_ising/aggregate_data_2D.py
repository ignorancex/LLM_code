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

		L_dir = os.path.join(data_dir, J, "L{:d}".format(L))
		output_dir = os.path.join(L_dir, "aggregate")

		states = np.load(os.path.join(output_dir, "states.npz"))

		states_train = states["train"].reshape((states["train"].shape[0], states["train"].shape[1], L, L))
		states_train_symmetric = 4/L**2 * np.stack([ \
			np.sum(states_train[:,:,::2,::2], (2, 3)), \
			np.sum(states_train[:,:,::2,1::2], (2, 3)), \
			np.sum(states_train[:,:,1::2,::2], (2, 3)), \
			np.sum(states_train[:,:,1::2,1::2], (2, 3)) \
		], 2)
		states_test = states["test"].reshape((states["test"].shape[0], states["test"].shape[1], L, L))
		states_test_symmetric = 4/L**2 * np.stack([ \
			np.sum(states_test[:,:,::2,::2], (2, 3)), \
			np.sum(states_test[:,:,::2,1::2], (2, 3)), \
			np.sum(states_test[:,:,1::2,::2], (2, 3)), \
			np.sum(states_test[:,:,1::2,1::2], (2, 3)) \
		], 2)
		np.savez(os.path.join(output_dir, "states_symmetric_2D.npz"), train=states_train_symmetric, test=states_test_symmetric)


print("All done!")
