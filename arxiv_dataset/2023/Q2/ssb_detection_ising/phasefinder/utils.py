import os
import numpy as np
import torch

def inverse_softplus(y):
	return y + torch.log(1.0 - torch.exp(-y))


def nullproj(A):
	I = torch.eye(A.shape[1], dtype=A.dtype)
	return I - torch.linalg.lstsq(A, A, driver="gelsd").solution


def build_path(results_dir, J, observable_name, L, N=None, fold=None, seed=None, L_test=None, subdir=None):
	if subdir is not None:
		results_dir = os.path.join(results_dir, subdir)
	path = os.path.join(results_dir, J, observable_name)
	if L is not None:
		path = os.path.join(path, "L{:d}".format(L))
	if observable_name == "magnetization":
		return path
	path = os.path.join(path, "N{:d}".format(N), "fold{:d}".format(fold), "seed{:d}".format(seed))
	if L_test is None or L_test == L:
		return path
	path = os.path.join(path, "L{:d}".format(L_test))
	return path
