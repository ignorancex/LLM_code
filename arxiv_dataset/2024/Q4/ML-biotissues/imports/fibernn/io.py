import adios2
import torch
import numpy as np
from . import mechanics
from torch.utils import data
from matplotlib import pyplot as plt
from torch import Tensor


def _read_adios2(key, filename) -> np.ndarray:
    with adios2.open(filename, "r") as fh:
        for step in fh:
            step_vars = step.available_variables()
            data = step.read(key)
            return data


class Adios2NetworkDataset(torch.utils.data.Dataset):
    def __init__(self, filename, dtype=torch.float32):
        """
        path to adios2 output data
        """
        # Since the neohookean energy function is based on C/U rather than F, I construct random U rather
        # than random F. The reduces the number of samples to sets.
        print('Reading adios files...')
        stiffness = _read_adios2("stiffness", filename)
        stiffness_evals = np.linalg.eigvals(stiffness)
        bad_mask = np.min(stiffness_evals, axis=1) < 0

        self.stiffness = torch.from_numpy(stiffness[~bad_mask]).to(dtype)
        self.deformation_gradient = torch.from_numpy(
            _read_adios2("F", filename)[~bad_mask]).to(dtype)
        self.right_cauchy_green = mechanics.compute_right_cauchy_green(self.deformation_gradient)
        # self.right_cauchy_green = self.right_cauchy_green.to(torch.float32)
        self.energy = torch.from_numpy(_read_adios2("strain_energy",
                                                    filename)[~bad_mask]).to(dtype)
        # print(self.energy)
        self.stress = torch.from_numpy(_read_adios2("stress", filename)[~bad_mask]).to(dtype)

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.deformation_gradient)

    def __getitem__(self, idx):
        return self.right_cauchy_green[idx], self.deformation_gradient[idx], self.energy[idx], self.stress[idx], self.stiffness[idx]


def dataset_to_array(dataset: data.DataLoader, dtype=torch.float32) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    target_energies = torch.zeros(len(dataset), requires_grad=False, dtype=dtype)
    right_cauchy_greens = torch.zeros((len(dataset), 3, 3), requires_grad=False, dtype=dtype)
    deformation_gradients = torch.zeros((len(dataset), 3, 3), requires_grad=False, dtype=dtype)
    target_cauchy_stresses = torch.zeros((len(dataset), 6), requires_grad=False, dtype=dtype)
    target_stiffnesses = torch.zeros((len(dataset), 6, 6), requires_grad=False, dtype=dtype)
    with torch.no_grad():
        for i, (right_cauchy_green, deformation_gradient, target_energy, target_stress, target_stiffness) in enumerate(dataset):
            target_energies[i] = target_energy
            right_cauchy_greens[i] = right_cauchy_green
            deformation_gradients[i] = deformation_gradient
            target_cauchy_stresses[i] = target_stress
            target_stiffnesses[i] = target_stiffness
    return right_cauchy_greens, deformation_gradients, target_energies, target_cauchy_stresses, target_stiffnesses


def tensor2ndarr(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class DataLib:
    def __init__(self, data_fname: str, seed: int) -> None:
        self.dat = Adios2NetworkDataset(data_fname)
        self.test, self.train = data.random_split(self.dat, [0.20, 0.80],
                                                  generator=torch.Generator().manual_seed(seed))
