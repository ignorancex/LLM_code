# This module defines classes for various analytical functions used in optimization

import numpy as np

class Ackley():

    def __init__(self, dim):
        """
            Class for defining the ackley function

            Parameters
            ----------
            dim: int
                dimension of the function
        """

        self.dim = dim
        self.lb = -32.768*np.ones(dim)
        self.ub = 32.768*np.ones(dim)

        self.a = 20.0
        self.b = 0.2
        self.c = 2.0*np.pi

    def __call__(self, x):

        return -self.a * np.exp( -self.b * np.linalg.norm(x, axis=-1) / np.sqrt(self.dim) ) \
            - np.exp( np.mean(np.cos(self.c*x), axis=-1) ) + self.a + np.e

class Levy():

    def __init__(self, dim):
        """
            Class for defining the levy function

            Parameters
            ----------
            dim: int
                dimension of the function
        """

        self.dim = dim
        self.lb = -10.0*np.ones(dim)
        self.ub = 10.0*np.ones(dim)

    def __call__(self, x):

        w = 1.0 + (x - 1.0) / 4.0

        part1 = np.sin(np.pi*w[:,0])**2
        
        part2 = np.sum( (w[:,:-1] - 1.0)**2 * (1.0 + 10.0*np.sin(np.pi*w[:,:-1] + 1.0)**2), axis=-1 )

        part3 = (w[:,-1] - 1.0)**2 * (1.0 + np.sin(2*np.pi*w[:,-1])**2 )

        return part1 + part2 + part3

class Rastrigin():

    def __init__(self, dim):
        """
            Class for defining the rastrigin function

            Parameters
            ----------
            dim: int
                dimension of the function
        """

        self.dim = dim
        self.lb = -5.12*np.ones(dim)
        self.ub = 5.12*np.ones(dim)

    def __call__(self, x):

        return 10.0*self.dim + np.sum(x**2 - 10.0*np.cos(2.0*np.pi*x), axis=-1)
    
class Hartmann6D():

    def __init__(self):
        """
            Class for defining the Hartmann 6D function

            Parameters
            ----------
            dim: int
                dimension of the function
        """

        self.dim = 6
        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def __call__(self, x):

        # P matrix
        P = np.array([[1312, 1696, 5569, 124, 8283, 5886],
                        [2329, 4135, 8307, 3736, 1004, 9991],
                        [2348, 1451, 3522, 2883, 3047, 6650],
                        [4047, 8828, 8732, 5743, 1091, 381]])

        # A matrix
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                        [0.05, 10, 17, 0.1, 8, 14],
                        [3, 3.5, 1.7, 10, 17, 8],
                        [17, 8, 0.05, 10, 0.1, 14]])
        
        # alpha vector
        alpha = np.array([[1], [1.2], [3.0], [3.2]])

        innersum = np.sum(A * (np.expand_dims(x,-2) - 1e-4 * P)**2, axis=-1)

        y = -np.matmul(np.exp(-innersum), alpha)

        return y
