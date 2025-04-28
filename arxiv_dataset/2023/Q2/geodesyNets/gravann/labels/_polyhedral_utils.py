import numpy as np


def calculate_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Calculates the volume for a given polyhedron consisting of vertices and triangular faces.
    The vertices need to be consistent in clockwise or anticlockwise fashion!

    The volume of polyhedron consisting of triangles can be calculated by summing up the
    determinant of each 3x3 matrix (consisting of the 3 corners) and dividing the result by 6.

    Args:
        vertices: an (N, 3) array-like
        faces: an (M, 3) array-like

    Returns:
        the volume of the polyhedron

    References:
        PHILIP J. SCHNEIDER, DAVID H. EBERLY, in Geometric Tools for Computer Graphics, 2003
        (Chapter 13.12.3 Volume of Polyhedron)
    """
    faces_resolved = np.array([[vertices[x], vertices[y], vertices[z]] for x, y, z in faces])
    return np.abs(np.linalg.det(faces_resolved).sum() / 6.0)


def calculate_density(vertices: np.ndarray, faces: np.ndarray, mass: float = 1.0) -> float:
    """
    Calculates the density for a given polyhedron consisting of vertices and triangular faces.
    Args:
        vertices: an (N, 3) array-like
        faces: an (M, 3) array-like
        mass: the mass of the polyhedron (default: 1.0)

    Returns:
        the density of the polyhedron
    """
    return mass / calculate_volume(vertices, faces)
