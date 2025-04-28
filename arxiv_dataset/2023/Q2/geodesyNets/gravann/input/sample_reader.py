import pickle as pk

import numpy as np
import torch


def load_sample(sample: str, use_differential: bool = False) -> torch.Tensor:
    """Loads the mascon model of the sample

    Args:
        sample: Sample to load
        use_differential: if true, loads the non-uniform mascon data from the file

    Returns:
        points and masses of the sample
    """

    with open("mascons/" + sample, "rb") as file:
        mascon_points, mascon_masses_u, name = pk.load(file)

    mascon_points = torch.tensor(mascon_points)
    mascon_masses_u = torch.tensor(mascon_masses_u)

    if use_differential:
        try:
            with open("mascons/" + sample[:-3] + "_nu.pk", "rb") as file:
                _, mascon_masses_nu, _ = pk.load(file)
            mascon_masses_nu = torch.tensor(mascon_masses_nu)
            print("Loaded non-uniform model")
        except:
            mascon_masses_nu = None
    else:
        mascon_masses_nu = None

    # If we are on the GPU , make sure these are on the GPU. Some mascons were stored as tensors on the CPU. it is weird.
    if torch.cuda.is_available():
        mascon_points = mascon_points.cuda()
        mascon_masses_u = mascon_masses_u.cuda()
        if mascon_masses_nu is not None:
            mascon_masses_nu = mascon_masses_nu.cuda()

    print("Name: ", name)
    print("Number of mascon_points: ", len(mascon_points))
    print("Total mass: ", sum(mascon_masses_u).item())
    return mascon_points, mascon_masses_u, mascon_masses_nu


def load_polyhedral_mesh(sample: str, resolution: str = "100%") -> (np.ndarray, np.ndarray):
    """Loads a polyhedral mesh for a given sample from the '3dmeshes' folder.

    Args:
        sample: the name of file/ sample
        resolution: either '100%' (normal mesh, default value) or '10%' (low-poly) or '1%' or '0.1%'

    Returns:
        tuple of vertices (N, 3), triangles (M, 3)

    """
    suffix = {
        "100%": "",
        "10%": "_lp",
        "1%": "_llp",
        "0.1%": "_lllp"
    }
    with open(f"./3dmeshes/{sample}{suffix[resolution]}.pk", "rb") as f:
        vertices, triangles = pk.load(f)
        return np.array(vertices), np.array(triangles)


def load_mascon_data(sample: str) -> (torch.Tensor, torch.Tensor):
    """Loads the mascon points and mascon masses for a given sample from the 'mascons' folder

    Args:
        sample: the name of the file/ sample

    Returns:
        tuple of mascon_points (N, 3), mascon_masses (N)

    """
    with open(f"./mascons/{sample}.pk", "rb") as file:
        mascon_points, mascon_masses_u, name = pk.load(file)
        mascon_points = torch.tensor(mascon_points)
        mascon_masses_u = torch.tensor(mascon_masses_u)
        return mascon_points, mascon_masses_u
