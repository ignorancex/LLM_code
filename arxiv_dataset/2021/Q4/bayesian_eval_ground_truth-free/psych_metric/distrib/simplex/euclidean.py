"""The functions for converting to and from the Euclidean simplex basis"""
from copy import deepcopy
import logging

import numpy as np


# TODO simplex transform class, euclidean child, hyperbolic child
# class is really just a struct and convenience of streamlining process

#def get_simplex_basis_matrix(input_dim):
def get_change_of_basis_matrix(input_dim):
    """Creates a matrix that transforms from an `input_dim` space
    representation of a `input_dim` - 1 discrete probability simplex to
    that probability simplex's space.

    The first dimension is arbitrarily chosen to collapse and is the
    expected dimension to be adjusted for when moving the origin in the
    change of basis.

    Parameters
    ----------
    input_dim : int
        The dimensions of the space of the data of a discrete probability
        distribution. The valid values in this `input-dim`-space can be
        expressed as a probability simplex where all points in that space
        can be expressed as vectors whose values sum to 1 and are all in
        the range [0,1].

    Returns
    -------
    np.ndarray
        Invertable matrix that is used to convert from the
        `input_dim`-space to the probabilistic simplex space as a
        left-transfrom matrix and converts the other way as a
        right-transform matrix.
    """
    # Create n-1 spanning vectors, Using first dim as origin of new basis.
    spanning_vectors = np.vstack((
        -np.ones(input_dim - 1),
        np.eye(input_dim - 1),
    ))

    # TODO to transpose or not to transpose... just ensure you're consistent!
    # Create orthonormal basis of simplex
    return np.linalg.qr(spanning_vectors)[0]


def transform_to(vectors, transform_matrix, origin_adjust):
    """Transforms the vectors from discrete distribtuion space into the
    probability simplex space of one dimension less. The orgin adjustment
    is used to move to the correct origin of the probability simplex space.
    """
    return (vectors - origin_adjust) @ transform_matrix


def transform_from(vectors, transform_matrix, origin_adjust):
    return (vectors @ transform_matrix.T) + origin_adjust


class EuclideanSimplexTransform(object):
    """Creates and contains the objects needed to convert to and from the
    Euclidean Simplex basis.

    Attributes
    ----------
    origin_adjust : np.ndarray
    change_of_basis_matrix : np.ndarray

    Properties
    ----------
    input_dim : int
        The number of dimensions of the input samples before being transformed.
    output_dim : int
        The number of dimensions of the samples after being transformed.
    """
    def __init__(self, input_dim, equal_scale=True):
        # Create origin adjustment, center 1st dim's extreme value at origin
        self.origin_adjust = np.zeros(input_dim)
        self.origin_adjust[0] = 1

        # Create change of basis matrix
        self.change_of_basis_matrix = get_change_of_basis_matrix(input_dim)

        #if equal_scale:
        # THIS IS NOT necessary for Euclidean transform.
        #    simplex_extremes = np.eye(input_dim) / self.to(np.eye(input_dim))
        #    self.change_of_basis_matrix =

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
        return len(self.origin_adjust)

    @property
    def output_dim(self):
        return self.input_dim - 1

    def to(self, vectors):
        """Transform given vectors into n-1 probability simplex space."""
        # center 1st dim's extreme value at origin
        return transform_to(
            vectors,
            self.change_of_basis_matrix,
            self.origin_adjust,
        )

    def back(self, vectors):
        """Transform given vectors out of n-1 probability simplex space."""
        return transform_from(
            vectors,
            self.change_of_basis_matrix,
            self.origin_adjust,
        )

    def tf_back(self, vectors):
        """Transform given vectors out of n-1 probability simplex space."""
        return (vectors @ tf.transpose(self.transform_matrix)) \
            + self.origin_adjust
