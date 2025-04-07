"""Example scripts for using the tdse module
"""
from typing import List, Dict, Tuple

from numpy import unique, array, concatenate, zeros
from scipy.sparse import dok_matrix
import qlp.mds.transformation as transform


def embed_qubo_example(n_vertices: int) -> Tuple[dok_matrix, Dict[int, List[int]]]:
    """Returns qubo and explicit embedding for NN(n_vertices) graph

    Only implemented for n_vertices=2
    """
    if n_vertices == 2:
        """This is a NN(2) graph embedded in Chimera
        Embedded QUBO form from DWave
        (1161, 1161): 6.5
        (1162, 1162): -3.0
        (1164, 1164): 6.0
        (1165, 1165): 6.5
        (1166, 1166): 6.0

        (1161, 1164): -4.0
        (1161, 1165): -16.0
        (1161, 1166): -4.0

        (1162, 1164): -4.0
        (1162, 1165): 8.0
        (1162, 1166): -4.0

        As matrix
        [[6.5     0  -4.0  -16.0  -4.0]
         [  0  -3.0  -4.0    8.0  -4.0]
         [  0     0   6.0      0     0]
         [  0     0     0    6.5     0]
         [  0     0     0      0  6.0]] 
        """
        q = [[6.5, 0, -4.0, -16.0, -4.0],
             [0, -3.0, -4.0, 8.0, -4.0],
             [0, 0, 6.0, 0, 0],
             [0, 0, 0, 6.5, 0],
             [0, 0, 0, 0, 6.0]]
        embedding = dict()
        embedding[0] = [1161, 1165]
        embedding[1] = [1162]
        embedding[2] = [1166]
        embedding[3] = [1164]
    else:
        raise ValueError("No embedded graph defined.")

    dok_qubo = dok_matrix(q)

    return dok_qubo, embedding
