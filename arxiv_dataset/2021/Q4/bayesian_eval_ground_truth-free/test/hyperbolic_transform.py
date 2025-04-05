"""Unit tests for all parts of the Hyperbolic transform that goes from the
probability simplex to its hyperbolic embedding.
"""
import numpy as np

from psych_metric.distrib.simplex import hyperbolic

# TODO rotate_around() test

# TODO rotator test
def rotator_test():
    """Tests the Rotator
    """
    #TODO ensure edge lengths between all simplex vertices are preserved
    #TODO ensure the zeroed out dim is zeroed out. isclose()
    #TODO ensure able to

# TODO test Euclidean Simplex Transform (rotate and center)
#   TODO test compare results of @ cart_simplex and rotator.rotate()

# TODO test get_simplex_boundary_pts()

# TODO test the cart_simplex to hypersphere? problem: its embedded in hyperbolic simplex transform

# TODO test Hyperbolic Simplex Transform
