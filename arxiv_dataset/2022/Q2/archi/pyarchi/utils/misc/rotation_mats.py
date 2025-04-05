import numpy as np
from scipy import optimize


def matrix_cnter_clock(angle):
    """
    Creates a rotation matrix for counter clockwise rotations.
    Expects an angle in degrees

    :param angle: rotation angle, in degrees.
    :return: counter clockwise rotation matrix
    """
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def matrix_clockwise(angle):
    """
    Creates a rotation matrix for clockwise rotations.
    Expects an angle in degrees

    :param angle: rotation angle, in degrees.
    :return: counter clockwise rotation matrix
    """
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


if __name__ == "__main__":
    pass
