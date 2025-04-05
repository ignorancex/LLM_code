import os
import time
import itertools
import numpy as np

from phasefinder.datasets import Ising


time_start = time.time()

Js = [(1, "ferromagnetic"), (-1, "antiferromagnetic")]
Ls = [16, 32, 64, 128]
Ts = [1.04+0.04*i for i in range(25)] \
	+ [2.01+0.01*i for i in range(50)] \
	+ [2.54+0.04*i for i in range(25)]

for ((J, J_long), L, T) in itertools.product(Js, Ls, Ts):
	os.makedirs("data/{}/L{:d}".format(J_long, L), exist_ok=True)

	I = Ising()
	I.generate(d=2, 
		L=L, 
		J=J, 
		T=T, 
		mc_steps=50000, 
		ieq_steps=10000, 
		meas_steps=10, 
		seed=107, 
		output_dir="data/{}/L{:d}/T{:.2f}".format(J_long, L, T), 
		encode=True)

	I.reduce_checkerboard("data/{}/L{:d}/T{:.2f}".format(J_long, L, T), decode=True)


print("Done!")
print("Took {:.3f} seconds.".format(time.time()-time_start))
