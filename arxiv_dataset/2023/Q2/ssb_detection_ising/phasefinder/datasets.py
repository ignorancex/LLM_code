import os
import json
import time
import numpy as np
import pandas as pd
from scipy.linalg import circulant

from . import compressor


class Ising(object):

	def __init__(self):
		pass

	def generate(self, d=2, L=20, J=1, T=4.0, mc_steps=10000, ieq_steps=1000, meas_steps=5, seed=107, ising_program="../ising/install/bin/ising", output_dir="output", encode=False):
		time_start = time.time()
		kwargs = locals()
		kwargs.pop("self")
		for (name, value) in kwargs.items():
			setattr(self, name, value)

		assert not os.path.isdir("output"), "The directory \"output\" already exists and cannot be overwritten."
		assert J==1 or J==-1, "J must be either 1 (ferromagnetic) or -1 (antiferromagnetic; got {:d} instead.".format(J)

		os.system("{} -d {:d} -L {:d} -J {:d} -T {:f} --nmcs {:d} --ieq {:d} --nmeas {:d} -s {:d}".format(ising_program, d, L, J, T, mc_steps, ieq_steps, meas_steps, seed))
		if output_dir != "output":
			os.system("mv output {}".format(output_dir))
		if encode:
			with open(os.path.join(output_dir, "states.txt"), "r") as fp:
				states_str = fp.read()
			states_enc = compressor.encode(states_str)
			with open(os.path.join(output_dir, "states.enc"), "w") as fp:
				fp.write(states_enc)
			os.system("rm {}".format(os.path.join(output_dir, "states.txt")))

		with open(os.path.join(output_dir, "time.txt"), "w") as fp:
			fp.write("{:f}".format(time.time()-time_start))

	def reduce_checkerboard(self, data_dir, decode=False):
		time_start = time.time()
		states = self.load_states(data_dir, decode=decode)
		assert self.d == 2, "Checkerboard representation is supported only for dimension d = 2."
		states_symmetric = 2/self.L**2 * np.stack([ \
			np.sum(states[:,::2,::2]+states[:,1::2,1::2], (1, 2)), \
			np.sum(states[:,1::2,::2]+states[:,::2,1::2], (1, 2)) \
		], 1)

		np.save(os.path.join(data_dir, "states_symmetric.npy"), states_symmetric)

		with open(os.path.join(data_dir, "time_symmetric.txt"), "w") as fp:
			fp.write("{:f}".format(time.time()-time_start))

	def load_args(self, data_dir):
		with open(os.path.join(data_dir, "args.json"), "r") as fp:
			args = json.load(fp)
		self.d = args["d"]
		self.L = args["L"]
		self.J = args["J"] if "J" in args.keys() else 1
		self.T = args["T"]
		self.mc_steps = args["nmcs"]
		self.ieq_steps = args["ieq"]
		self.meas_steps = args["nmeas"]
		self.seed = args["s"]

	def load_states(self, data_dir, decode=False, n_samples=None, dtype=None, flatten=False, channel_dim=False, symmetric=False):
		self.load_args(data_dir)

		if symmetric:
			states = np.load(os.path.join(data_dir, "states_symmetric.npy"))

		else:

			filename = "states.enc" if decode else "states.txt"
			with open(os.path.join(data_dir, filename), "r") as fp:
				states = fp.read()
			if decode:
				states = compressor.decode(states)

			states = 2*np.array([char for char in states], dtype=int)-1
			assert not (flatten and channel_dim), "Cannot add channel dimension if also flattening."
			if flatten:
				dimension = self.L**self.d
				states = states.reshape((-1, dimension))
			else:
				dimensions = [self.L]*self.d
				if channel_dim:
					dimensions = [1] + dimensions
				states = states.reshape((-1, *dimensions))

		if n_samples is not None:
			assert n_samples <= states.shape[0], "There are only {:d} samples in the dataset.".format(states.shape[0])
			states = states[-n_samples:]

		if dtype is not None:
			states = states.astype(dtype)

		return states

	def load_M(self, data_dir, per_spin=False):
		self.load_args(data_dir)
		Ms = pd.read_csv(os.path.join(data_dir, "EMR.csv"))["M"].values
		if per_spin:
			Ms = Ms/self.L**2

		return Ms

	def magnetization(self, data_dir, per_spin=False, staggered=False):
		self.load_args(data_dir)
		reduced_states = np.load(os.path.join(data_dir, "states_symmetric.npy"))

		if staggered:
			Ms = (reduced_states[:,0] - reduced_states[:,1])/2
		else:
			Ms = reduced_states.sum(1)/2

		if not per_spin:
			Ms = self.L**2 * Ms

		return Ms

	@staticmethod
	def jackknife(samples, func, samples_2=None):
		estimate = func(samples.reshape((1, -1))).sum()
		jk_estimates = func(np.flip(circulant(np.flip(samples)).T, 1)[:,1:])
		if samples_2 is not None:
			estimate_2 = func(samples_2.reshape((1, -1))).sum()
			jk_estimates_2 = func(np.flip(circulant(np.flip(samples_2)).T, 1)[:,1:])
			estimate = estimate/estimate_2
			jk_estimates = jk_estimates[:,None]/jk_estimates_2[None,:]
		error = np.sqrt(np.sum((jk_estimates-estimate)**2))
		return estimate, error
