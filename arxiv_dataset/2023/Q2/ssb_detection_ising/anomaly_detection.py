import os
import time
import itertools
import warnings
warnings.filterwarnings("ignore")  # due to no GPU

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import phasefinder as pf


### Abbreviated version of the Ising class from phasefinder.datasets adapted for the external field experiment

class Ising(object):

	def __init__(self):
		pass

	def generate(self, d=2, L=20, J=1, B=0.0, T=4.0, mc_steps=10000, ieq_steps=1000, meas_steps=5, seed=107, ising_program="../ising/install/bin/ising", output_dir="output", encode=False):
		time_start = time.time()
		kwargs = locals()
		kwargs.pop("self")
		for (name, value) in kwargs.items():
			setattr(self, name, value)

		assert J==1 or J==-1, "J must be either 1 (ferromagnetic) or -1 (antiferromagnetic; got {:d} instead.".format(J)

		os.system("{} -d {:d} -L {:d} -J {:d} -B {:f} -T {:f} --nmcs {:d} --ieq {:d} --nmeas {:d} -s {:d}".format(ising_program, d, L, J, B, T, mc_steps, ieq_steps, meas_steps, seed))
		os.system("mv output_B{:1.3f}_T{:1.2f} {}".format(B, T, output_dir))

		with open(os.path.join(output_dir, "states.txt"), "r") as fp:
			states = fp.read()
		states = np.array([char for char in states]).reshape((-1, L**2))
		states = 2*states.astype(np.float32)-1
		np.save(os.path.join(output_dir, "states.npy"), states)

		os.system("rm {}".format(os.path.join(output_dir, "states.txt")))

		states = states.reshape((states.shape[0], L, L))
		states_symmetric = 2/L**2 * np.stack([ \
			np.sum(states[:,::2,::2]+states[:,1::2,1::2], (1, 2)), \
			np.sum(states[:,1::2,::2]+states[:,::2,1::2], (1, 2)) \
		], 1)
		np.save(os.path.join(output_dir, "states_symmetric.npy"), states_symmetric)

		with open(os.path.join(output_dir, "time.txt"), "w") as fp:
			fp.write("{:f}".format(time.time()-time_start))


### Simulation of the Ising model in an external field

time_start = time.time()
print("Simulating Ising model . . . ")

Bs = [0.0, 0.001, 0.01, 0.1]
Ts = [2.0, 2.5]

for (B, T) in itertools.product(Bs, Ts):
	os.makedirs("data/anomaly_detection/L128/B{:.3f}".format(B), exist_ok=True)

	I = Ising()
	I.generate(d=2, 
		L=128, 
		J=1, 
		B=B,
		T=T, 
		mc_steps=25000, 
		ieq_steps=10000, 
		meas_steps=10, 
		seed=107, 
		output_dir="data/anomaly_detection/L128/B{:.3f}/T{:.2f}".format(B, T), 
		encode=False)

print("Finished all simulations")
print("Took {:.3f} seconds.".format(time.time()-time_start))


### Evaluate trained encoders on simulated data in an external field

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

print("Evaluating encoders . . . ")

means_0 = np.zeros((8, 3))

file = open("results/processed/anomaly_detection_FM.csv", "w")
file.write("temperature,field_strength,baseline_mean,baseline_std,ge_mean,ge_std\n")

for (T, B) in itertools.product(Ts, Bs):
	for model in ["latent", "latent_equivariant"]:
		model_name = "baseline" if model=="latent" else "GE"
		print("T={:1.2f}, B={:1.3f}, {}".format(T, B, model_name))
		input_dim = 2 if model=="latent_equivariant" else 128**2
		filename = "states_symmetric.npy" if model=="latent_equivariant" else "states.npy"
		encoder = pf.models.MLP(input_dim, 4, 1)
		decoder = pf.models.MLP(2, 64, input_dim)
		dists = []
		for (fold, seed) in itertools.product(range(8), range(3)):
			dir = "results/ferromagnetic/{}/L128/N256/fold{:d}/seed{:d}".format(model, fold, seed)
			trainer = pf.trainers.Autoencoder(encoder, decoder)
			trainer.load_encoder(os.path.join(dir, "encoder.pth"))
			dataX = np.load("data/anomaly_detection/L128/B{:1.3f}/T{:1.2f}/{}".format(B, T, filename))
			dataX = dataX[500:]
			dataT = np.full((dataX.shape[0], 1), T, dtype=np.float32)
			data_loader = DataLoader(TensorDataset(torch.as_tensor(dataX), torch.as_tensor(dataT)), batch_size=2500, shuffle=False, drop_last=False, num_workers=8)
			encodings = trainer.encode(data_loader)
			if B < 1e-9:
				means_0[fold, seed] = encodings.mean()
			dist = np.abs( encodings.mean()-means_0[fold, seed] )
			dists.append(dist)
		dists = np.array(dists)
		mean = dists.mean()
		std = dists.std()
		if std > 0:
			mean = mean/std
			samples = np.random.choice(dists, size=(10000, 24))
			std = ( samples.mean(1)/samples.std(1) ).std()
		if model == "latent":
			file.write("{:1.2f},{:1.3f},{:.5f},{:.5f},".format( \
				T, B, mean, std))
		else:
			file.write("{:.5f},{:.5f}\n".format( \
				mean, std))

file.close()

print("All done!")
