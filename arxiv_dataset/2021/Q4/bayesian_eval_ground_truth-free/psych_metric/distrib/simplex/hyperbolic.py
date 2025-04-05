"""The functions for converting to and from the Hyperbolic simplex basis"""
from copy import deepcopy
import logging

import numpy as np

#from psych_metric.distrib.simplex.euclidean import EuclideanSimplexTransform


def cart2polar(vectors):
    """Convert from 2d Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    vectors : np.ndarray
        2-dimensional array where the first dimension is the samples and the
        second is the 2-dimensional Cartesian coordinates.

    Returns
    -------
        2-dimensional array where the first dimension is the samples and the
        second dimension contains 2 elements consisting of the polar
        coordinates where the first element is the radius, and then followed by
        the angle.
    """
    return np.concatenate(
        (
            np.linalg.norm(vectors, axis=1, keepdims=True),
            np.arctan2(vectors[:, 1], vectors[:, 0]).reshape([-1, 1]),
        ),
        axis=1,
    )


def polar2cart(vectors):
    """Convert from polar to 2d Cartesian coordinates.

    Parameters
    ----------
    vectors : np.ndarray
        2-dimensional array where the first dimension is the samples and the
        second dimension contains 2 elements consisting of the polar
        coordinates where the first element is the radius, and then followed by
        the angle.

    Results
    -------
    np.ndarray
        2-dimensional array where the first dimension is the samples and the
        second is the 2-dimensional Cartesian coordinates.
    """
    return vectors[:, [0]] * np.concatenate(
        (
            np.cos(vectors[:, [1]]),
            np.sin(vectors[:, [1]]),
        ),
        axis=1,
    )


def cartesian_to_hypersphere(vectors):
    """Convert from Cartesian coordinates to hyperspherical coordinates of the
    same n-dimensions.

    Parameters
    ----------
    vectors : np.ndarray
        2-dimensional array where the first dimension is the samples and the
        second is the n-dimensional Cartesian coordinates.

    Results
    -------
    np.ndarray
        2-dimensional array where the first dimension is the samples and the
        second dimension contains n elements consisting of the n-1-dimensional
        hyperspherical coordinates where the first element is the radius, and
        then followed by n-1 angles for each dimension.
    """
    if len(vectors.shape) == 1:
        # single sample
        vectors = vectors.reshape([1, -1])

    if vectors.shape[1] == 2:
        return cart2polar(vectors)
    elif vectors.shape[1] < 2:
        raise ValueError(' '.join([
            'Expected the number of coordinate dimensions to be >= 2, but',
            f'recieved vectors with shape {vectors.shape}. and axis being',
            f'1.',
        ]))

    #radii = np.linalg.norm(vectors[:, 0])
    flipped = np.fliplr(vectors)
    cumsqrt = np.sqrt(np.cumsum(flipped ** 2, axis=1))
    #radii = cumsqrt[:, -1]
    angles = np.arccos(flipped / cumsqrt)
    # angles 1 -- n-2 = np.fliplr(angles[2:])

    last_angle = np.pi - 2 * np.arctan(
        (flipped[:, 1] + cumsqrt[:, 1]) / flipped[:, 0]
    )

    # radius followed by ascending n-1 angles per row
    return np.concatenate(
        (
            cumsqrt[:, [-1]],
            np.fliplr(angles[:, 2:]),
            last_angle.reshape([-1, 1]),
        ),
        axis=1,
    )


def hypersphere_to_cartesian(vectors):
    """Convert from hyperspherical coordinates to Cartesian coordinates of the
    same n-dimensions.

    Parameters
    ----------
    vectors : np.ndarray
        2-dimensional array where the first dimension is the samples and the
        second dimension contains n elements consisting of the n-1-dimensional
        hyperspherical coordinates where the first element is the radius, and
        then followed by n-1 angles for each dimension.

    Results
    -------
    np.ndarray
        2-dimensional array where the first dimension is the samples and the
        second is the n-dimensional Cartesian coordinates.
    """
    if len(vectors.shape) == 1:
        # single sample
        vectors = vectors.reshape([1, -1])

    if vectors.shape[1] == 2:
        return polar2cart(vectors)
    elif vectors.shape[1] < 2:
        raise ValueError(' '.join([
            'Expected the number of coordinate dimensions to be >= 2, but',
            f'recieved vectors with shape {vectors.shape}. and axis being',
            f'1.',
        ]))

    # x1 = radius * cos(rho_1)
    # xn-1 = radius * sin(rho_1) * ... * sin(rho_n-2) * cos(rho_n-1)
    # xn = radius * sin(rho_1) * ... * sin(rho_n-1)
    sin = np.concatenate(
        (
            np.ones([vectors.shape[0], 1]),
            np.cumprod(np.sin(vectors[:, 1:]), axis=1)
        ),
        axis=1,
    )
    cos = np.concatenate(
        (np.cos(vectors[:, 1:]), np.ones([vectors.shape[0], 1])),
        axis=1,
    )
    return  vectors[:, 0].reshape(-1, 1) * sin * cos


def givens_rotation(dim, x, y, angle):
    """Creates a transposed Givens rotation matrix."""
    rotate = np.eye(dim)
    rotate[x, x] = np.cos(angle)
    rotate[x, y] = np.sin(angle)
    rotate[y, x] = -np.sin(angle)
    rotate[y, y] = np.cos(angle)
    return rotate


def rotate_around(rotation_simplex, angle):
    """General n-dimension rotation."""
    if rotation_simplex.shape[0] > rotation_simplex.shape[0]:
        # expects points to contained in the rows, elements in columns
        rotation_simplex = rotation_simplex.T

    # Transpose the simplex st the first point is centered at origin
    translation_vector = rotation_simplex[0].copy()
    v = rotation_simplex - translation_vector

    n = rotation_simplex.shape[1]
    mat = np.eye(n)

    k = 0

    for r in range(1, n-1):
        for c in list(range(r, n))[::-1]:
            k += 1
            rot = givens_rotation(
                n,
                c,
                c - 1,
                np.arctan2(v[r, c], v[r, c - 1]),
            )
            v = v @ rot
            mat = mat @ rot

    return (
        translation_vector,
        mat @ givens_rotation(n, n - 2, n - 1, angle) @ np.linalg.inv(mat),
    )


def get_simplex_boundary_pts(prob_vectors, copy=True):
    """Returns the boundary points of the regular simplex whose circumscribed
    hypersphenre's radius intersects through the provided points in Barycentric
    coordinates. The given points define the angle of the line that passes
    through the center of the simplex, the given point, and the respectie point
    on the boundary of the simplex.

    Parameters
    ----------
    prob_vectors : np.ndarray
        Array of probability vectors, or Barycentric coordinates of a regular
        simplex. Each point defines the angle of the ray that intersects the
        given point, starts at the center of the simplex, and intersects the
        corresponding boundary point of the simplex.
    copy : bool
        If True, copies the given prob_vectors np.ndarray, otherwise modifies
        the original.
    """
    if copy:
        prob_vectors = prob_vectors.copy()

    # Probability vectors are already in Barycentric coordinates
    # select minimum coord(s) as dim to zero to get boundary pt on d simplex
    row_min = np.min(prob_vectors, axis=1)
    dim = prob_vectors.shape[1] - 1

    for i, minimum in enumerate(row_min):
        min_mask = prob_vectors[i] == minimum
        prob_vectors[i, np.logical_not(min_mask)] += minimum / dim * min_mask.sum()
        prob_vectors[i, min_mask] = 0

    return prob_vectors


class Rotator(object):
    """Class the contains the process for rotating about some n-2 space."""
    def __init__(self, rotation_simplex, angle):
        self.translate, self.rotate_drop_dim = rotate_around(
            rotation_simplex,
            angle,
        )

    def rotate(self, vectors, drop_dim=False):
        """Rotates the vectors of the n-1 simplex from n dimensions to n-1
        dimensions.
        """
        # TODO expects shape of 2, add check on vectors
        result = (vectors - self.translate) @ self.rotate_drop_dim \
            + self.translate
        if drop_dim:
            return result[:, 1:]
        return result

    def inverse(self, vectors):
        """Rotates the vectors of the n-1 simplex from n-1 dimensions to n
        dimensions.
        """
        # TODO expects shape of 2, add check on vectors
        if vectors.shape[1] == len(self.translate):
            return (vectors - self.translate) \
                @ np.linalg.inv(self.rotate_drop_dim) + self.translate
        return (
            (
                np.hstack((np.zeros([len(vectors), 1]), vectors))
                - self.translate
            )
            @ np.linalg.inv(self.rotate_drop_dim)
            + self.translate
        )


class ProbabilitySimplexTransform(object):
    """Creates and contains the objects needed to convert to and from the
    Probability Simplex basis.

    Attributes
    ----------
    cart_simplex : np.ndarray
    centroid : np.ndarray
    rotator : Rotator
        Used to find cart_simplex and for reverse transforming from the
        cartesian simplex to the original probability simplex.

    Properties
    ----------
    input_dim : int
        The number of dimensions of the input samples before being transformed.
    output_dim : int

    Note
    ----
    This ProbabilitySimplexTransform takes more time and more memory than the
    original that used QR or SVD to find the rotation matrix. However, this
    version preserves the simplex dimensions, keeping the simplex regular,
    while the QR and SVD found rotation matrices do not.
    """
    def __init__(self, dim):
        prob_simplex_verts = np.eye(dim)

        # Get the angle to rotate about the n-2 space to zero out first dim
        angle_to_rotate = -np.arctan2(
            1.0,
            np.linalg.norm([1 / (dim - 1)] * (dim - 1)),
        )

        # Rotate to zero out one arbitrary dimension, drop that zeroed dim.
        self.rotator = Rotator(prob_simplex_verts[1:], angle_to_rotate)
        self.cart_simplex = self.rotator.rotate(prob_simplex_verts)

        # Center Simplex in (N-1)-dim (find centroid and adjust via that)
        self.centroid = np.mean(self.cart_simplex, axis=0)
        self.cart_simplex -= self.centroid

        # Save the vertices of the rotated simplex, transposed for ease of comp
        self.cart_simplex = self.cart_simplex.T
        # TODO Decide if keeping the cart_simplex for going from prob simplex
        # to cart simplex with only one matrix multiplication is worth keeping
        # the (n,n) matrix.

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        for k, v in self.__dict__.items():
            setattr(new, k, deepcopy(v, memo))

        return new

    @property
    def input_dim(self):
        # TODO wrap all code for obtaining cart_simllex in ProbabilityTransform
        #return self.euclid_simplex_transform.input_dim
        return self.cart_simplex.shape[0]

    @property
    def output_dim(self):
        #return self.euclid_simplex_transform.output_dim
        return self.cart_simplex.shape[1]

    def to(self, vectors, drop_dim=True):
        """Transform given vectors into hyperbolic probability simplex space."""
        # Convert from probability simplex to the Cartesian coordinates of the
        # centered, regular simplex
        if drop_dim:
            return (vectors @ self.cart_simplex)[:, 1:]
        return vectors @ self.cart_simplex

    def back(self, vectors):
        """Transform given vectors into hyperbolic probability simplex space."""
        # Convert from probability simplex to the Cartesian coordinates of the
        # centered, regular simplex
        #aligned = vectors @ self.aligned_simplex.T

        # TODO expects shape of 2, add check on vectors
        if vectors.shape[1] == self.input_dim:
            return self.rotator.inverse(vectors + self.centroid)
        return self.rotator.inverse(vectors + self.centroid)


class HyperbolicSimplexTransform(object):
    """

    Attributes
    ----------
    """

    def __init__(self, dim):
        self.pst = ProbabilitySimplexTransform(dim)

        # Save the radius of the simplex's circumscribed hypersphere
        self.circumscribed_radius = np.linalg.norm(self.pst.cart_simplex[:, 0])

    @property
    def input_dim(self):
        # TODO wrap all code for obtaining cart_simllex in ProbabilityTransform
        return self.pst.input_dim

    @property
    def output_dim(self):
        return self.pst.output_dim

    def to(self, vectors):
        """Transform given vectors into hyperbolic probability simplex space."""
        # Convert from probability simplex to the Cartesian coordinates of the
        # centered, regular simplex
        #aligned = vectors @ self.aligned_simplex.T
        aligned = self.pst.to(vectors, drop_dim=True)

        # Stretch simplex into hypersphere, no longer conserving the angles
        hyperspherical = cartesian_to_hypersphere(aligned)

        # Edge cases:
        #   Do not stretch points along line of vertex (all zeros but 1 element)
        #   Do not stretch centered points (all zeros).
        non_edge_case = (aligned == 0).sum(axis=1) < aligned.shape[1] - 1

        boundaries = get_simplex_boundary_pts(
            vectors[non_edge_case],
            copy=True,
        )

        # get boundary points radius
        boundary_radii = np.linalg.norm(
            self.pst.to(boundaries, drop_dim=True),
            axis=1,
        )

        # scale each point radii by * circum_radius / simplex_boundary_radius
        hyperspherical[non_edge_case, 0] = (
            hyperspherical[non_edge_case, 0]
            * self.circumscribed_radius / boundary_radii
        )

        # TODO Inverse Poincare' Ball method to go into hyperbolic space


        return hyperspherical

    def tf_to(self, vectors):
        """Transform given vectors into n-1 probability simplex space done in
        tensorflow code.
        """
        return

    def back(self, vectors):
        """Transform given vectors out of n-1 probability simplex space."""
        # Poinecare's Ball to get hypersphere
        #hyperspherical = poincare_ball(vectors)
        hyperspherical = vectors

        # Circumscribed hypersphere to Cartesian simplex:
        # vectors is the boundaries of the simplex, but in cart simplex.
        simplex = self.pst.back(hypersphere_to_cartesian(hyperspherical))
        non_edge_case = (simplex == 0).sum(axis=1) < simplex.shape[1] - 1

        # Get the boundaries in Barycentric coordinates (probability vectors)
        boundaries = get_simplex_boundary_pts(
            simplex[non_edge_case],
            copy=True,
        )

        # Get boundary points radius
        boundary_radii = np.linalg.norm(
            self.pst.to(boundaries, drop_dim=True),
            axis=1,
        )

        # Scale each point radii by * simplex_boundary_radius / circum_radius
        hyperspherical[non_edge_case, 0] = (
            hyperspherical[non_edge_case, 0]
            * boundary_radii / self.circumscribed_radius
        )

        # Cartesian simplex to probability distribution (Barycentric coord)
        return self.pst.back(hypersphere_to_cartesian(hyperspherical))

    def tf_from(self, vectors):
        """Transform given vectors out of n-1 probability simplex space done in
        tensorflow code.
        """
        return
