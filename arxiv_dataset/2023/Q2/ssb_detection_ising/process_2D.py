import os
import pickle
import itertools
import numpy as np
from phasefinder.utils import build_path


def calculate_generators_2D(results_dir, Js, encoder_names, Ls, Ns, folds, seeds):
	generator_types = ["rho", "tau", "sigma"]
	output_dict = {J: {name: {gen_type: {} for gen_type in generator_types} for name in encoder_names} for J in Js}
	for (J, name, gen_type) in itertools.product(Js, encoder_names, generator_types):
		gens = []
		for (L, N, fold, seed) in itertools.product(Ls, Ns, folds, seeds):
			with np.load(os.path.join(build_path(results_dir, J, name, L, N=N, fold=fold, seed=seed), "generator_reps.npz")) as fp:
				lams, V = np.linalg.eig(fp["sigma"])
				V = V[:,np.argsort(lams)]
				gens.append( np.linalg.inv(V)@fp[gen_type]@V )
		gens = np.stack(gens, 0)
		output_dict[J][name][gen_type]["mean"] = gens.mean(0)
		output_dict[J][name][gen_type]["std"] = gens.std(0)
	output_dir = os.path.join(results_dir, "processed")
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "generators_2D.pkl"), "wb") as fp:
		pickle.dump(output_dict, fp)


if __name__ == "__main__":
	results_dir = "results"
	Js = ["ferromagnetic", "antiferromagnetic"]
	Ls = [16, 32, 64, 128]
	Ns = [8, 16, 32, 64, 128, 256]

	folds = list(range(8))
	seeds = list(range(3))

	print("Calculating 2D symmetry generators . . . ")
	calculate_generators_2D(results_dir, Js, ["latent_equivariant_2D"], Ls, Ns, folds, seeds)

	print("Done!")
