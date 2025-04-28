from typing import Callable

import torch

from gravann.input import sample_reader
from . import mascon, polyhedral
from ._polyhedral_utils import calculate_density

_METHOD_REGISTRY = {
    ("mascon", True): mascon.acceleration,
    ("mascon", False): mascon.potential,
    ("polyhedral", True): polyhedral.acceleration,
    ("polyhedral", False): polyhedral.potential,
}


def bind_label(method: str, use_acc: bool = True, **kwargs) -> Callable[[torch.Tensor], torch.Tensor]:
    """Binds arguments to a given label function, so that it can be called by just inputting the tensor of points.

    Args:
        method: the method, either 'polyhedral' or 'mascon'
        use_acc: if to use acceleration for the evaluation
        **kwargs: arguments to bind

    Keyword Args:
        sample (str): utilized as path to the mesh/ mascon file. If given used to read in the mascon/ mesh data
        mascon_points (2-D array-like): an (N, 3) array-like object containing the points that belong to the mascon
        mascon_masses (1-D array-like): a (N,) array-like object containing the values for the uniform mascon masses.
                                        Can also be a scalar containing the mass value for all points.
        mesh_vertices (2-D array-like): an (N, 3) array-like object containing the vertices of the polyhedron
        mesh_faces (2-D array-like): a (N,) array-like object containing the edges of the polyhedron
        resolution (str): loads the given resolution for a polyhedral mesh if given. Per default 100%

    Returns:
        callable function taking an input tensor of points and returning the corresponding label's values

    """
    if method == 'mascon':
        if sample := kwargs.get("sample", None):
            mascon_points, mascon_masses = sample_reader.load_mascon_data(sample)
        else:
            mascon_points, mascon_masses = kwargs.get("mascon_points", None), kwargs.get("mascon_masses", None)
        return lambda points: _METHOD_REGISTRY[(method, use_acc)](points, mascon_points, mascon_masses)
    elif method == 'polyhedral':
        if sample := kwargs.get("sample", None):
            mesh_vertices, mesh_faces = sample_reader.load_polyhedral_mesh(sample, kwargs.get("resolution", "100%"))
        else:
            mesh_vertices, mesh_faces = kwargs.get("mesh_vertices", None), kwargs.get("mesh_faces", None)
        density = calculate_density(mesh_vertices, mesh_faces)
        return lambda points: _METHOD_REGISTRY[(method, use_acc)](points, mesh_vertices, mesh_faces, density)
    else:
        raise NotImplemented(f"The method {method} is not implemented!")
