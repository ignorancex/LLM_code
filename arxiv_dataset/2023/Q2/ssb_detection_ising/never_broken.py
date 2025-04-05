import itertools
import numpy as np
from gappy import gap

### custom GAP functions

@gap.gap_function
def group_from_generators(generators):
	"""
	function(generators)
	local G;
	generators := List(generators, PermList);
	G := GroupWithGenerators(generators);
	return G;
	end;
	"""

@gap.gap_function
def subgroup_generators(G, H):
	"""
	function(G, H)
	local generators;
	generators := List(GeneratorsOfGroup(H), g->Factorization(G, g));
	return generators;
	end;
	"""

def subgroup_generators_str(G, H, generator_names):
	generators = [g.__repr__() for g in subgroup_generators(G, H).python()]
	out = "["+", ".join(generators)+"]"
	for (i, name) in enumerate(generator_names, start=1):
		out = out.replace(f"x{i:d}", name)
	return out


### Ising symmetry group

class Ising(object):

	def __init__(self, L):
		self.L = L

		lat = np.arange(1, L**2+1).reshape((L, L))

		self.alpha = np.roll(lat, -1, axis=0).reshape((-1))
		self.alpha = np.concatenate([self.alpha, self.alpha+L**2], 0)

		self.rho = np.rot90(lat, k=-1).reshape((-1))
		self.rho = np.concatenate([self.rho, self.rho+L**2], 0)

		self.tau = np.flip(lat, axis=1).reshape((-1))
		self.tau = np.concatenate([self.tau, self.tau+L**2], 0)

		self.sigma = lat.reshape((-1))
		self.sigma = np.concatenate([self.sigma+L**2, self.sigma], 0)

		self.G = group_from_generators([list(gen) for gen in [self.alpha, self.rho, self.tau, self.sigma]])

	def print_subgroup(self, H):
		generator_names = ["alpha", "rho", "tau", "sigma"]
		return subgroup_generators_str(self.G, H, generator_names)


### subgroup of never-broken symmetries

def never_broken(G, degree):
	table = gap.CharacterTable(G)
	characters = gap.Irr(table)

	degrees = [gap.DegreeOfCharacter(char).python() for char in characters]
	indicators = gap.Indicator(table, 2).python()
	real_degrees = []
	for (deg, ind) in zip(degrees, indicators):
		if ind == 1:
			real_degrees.append(deg)
		else:
			real_degrees.append(2*deg)

	characters = [char for (char, deg) in zip(characters, real_degrees) if deg <= degree]
	kernels = [gap.KernelOfCharacter(char) for char in characters]
	return gap.Intersection(kernels)


if __name__ == "__main__":
	Ls = [4, 8, 16]
	degrees = [1, 2, 3, 4]

	for L in Ls:
		print(f"L = {L:d}:\n")
		print("Degree    |H|    H\n---")
		I = Ising(L)
		for degree in degrees:
			H = never_broken(I.G, degree)
			print(f"{degree:d}    {H.Order().python():d}    {I.print_subgroup(H)}")
		print()
