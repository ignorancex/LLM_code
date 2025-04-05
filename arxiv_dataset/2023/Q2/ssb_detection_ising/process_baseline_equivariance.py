import os
import json
import itertools
import numpy as np
from phasefinder.utils import build_path


def calculate_generators(results_dir, Js, encoder_names, Ls, Ns, folds, seeds):
	generator_types = ["alpha", "rho", "tau", "sigma"]
	output_dict = {J: {name: {gen_type: {} for gen_type in generator_types} for name in encoder_names} for J in Js}
	for (J, name, gen_type) in itertools.product(Js, encoder_names, generator_types):
		gens = []
		for (L, N, fold, seed) in itertools.product(Ls, Ns, folds, seeds):
			with open(os.path.join(build_path(results_dir, J, name, L, N=N, fold=fold, seed=seed), "cos_sims.json"), "r") as fp:
				gens.append( json.load(fp)[gen_type] )
		gens = np.array(gens)
		output_dict[J][name][gen_type]["mean"] = float(gens.mean())
		output_dict[J][name][gen_type]["std"] = float(gens.std())
	output_dir = os.path.join(results_dir, "processed")
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "baseline_equivariance.json"), "w") as fp:
		json.dump(output_dict, fp, indent=2)


if __name__ == "__main__":
	results_dir = "results"
	Js = ["ferromagnetic", "antiferromagnetic"]
	Ls = [16, 32, 64, 128]
	Ns = [8, 16, 32, 64, 128, 256]

	folds = list(range(8))
	seeds = list(range(3))

	print("Measuring baseline equivariance . . . ")
	calculate_generators(results_dir, Js, ["baseline_equivariance"], Ls, Ns, folds, seeds)

	print("Done!")
