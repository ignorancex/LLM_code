import polyhedral_gravity
import torch

from gravann.util.constants import GRAVITY_CONSTANT_INVERSE
from ._polyhedral_utils import calculate_density


def potential(target_points, mesh_vertices, mesh_faces, density=None):
    """Computes the gravity potential (G=1) created by a mascon in the target points.
    (to be used as Label in the training)

    Args:
        target_points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points where the potential is seeked.
        mesh_vertices (2-D array-like): an (N, 3) array-like object containing the vertices of the polyhedron
        mesh_faces (2-D array-like): a (N,) array-like object containing the edges of the polyhedron
        density (float, optional): the density of the polyhedron, if not given it is calculated from the given mesh

    Returns:
        1-D array-like: a (N, 1) torch tensor containing the gravity potential (G=1) at the target points
    """
    result = _evaluate_polyhedral_model(target_points, mesh_vertices, mesh_faces, density)
    try:
        return torch.tensor([pot for pot, acc, second_dev in result])
    except TypeError:
        # If target_points was a 1D array-like, handle it
        pot, acc, second_dev = result
        return torch.tensor([pot])


def acceleration(target_points, mesh_vertices, mesh_faces, density=None):
    """Computes the acceleration due to the mascon at the target points.
    (to be used as Label in the training)

    Args:
        target_points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points where the  acceleration should be computed.
        mesh_vertices (2-D array-like): an (N, 3) array-like object containing the vertices of the polyhedron
        mesh_faces (2-D array-like): a (N,) array-like object containing the edges of the polyhedron
        density (float, optional): the density of the polyhedron, if not given it is calculated from the given mesh

    Returns:
        1-D array-like: a (N, 3) torch tensor containing the acceleration (G=1) at the target points
    """
    result = _evaluate_polyhedral_model(target_points, mesh_vertices, mesh_faces, density)
    try:
        return torch.tensor([acc for pot, acc, second_dev in result])
    except TypeError:
        # If target_points was a 1D array-like, handle it
        pot, acc, second_dev = result
        return torch.tensor([acceleration])


def _evaluate_polyhedral_model(target_points, mesh_vertices, mesh_faces, density=None):
    if density is None:
        density = calculate_density(mesh_vertices, mesh_faces)
    density_gravity_factor = density * GRAVITY_CONSTANT_INVERSE * -1.0
    # Convert to numpy array, as the interface does not accept a torch tensor
    if torch.is_tensor(target_points):
        target_points = target_points.cpu().detach().numpy()
    return polyhedral_gravity.evaluate(mesh_vertices, mesh_faces, density_gravity_factor, target_points)
