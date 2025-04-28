import numpy as np
from laueotx.utils import logging as utils_logging
LOGGER = utils_logging.get_logger(__file__)


def divide_rodrigues_space(n_div):
    """
    Divides the rodrigues space in nDiv divisions
    TODO: add math reference
    """

    assert n_div>0
    
    n_div_cube = int(n_div**3)
    cubes = np.ones((n_div_cube, 3), dtype=PRECISION)

    for i in range(n_div_cube):

        d = i + 1
        cubes[i,0] = np.ceil(d/n_div**2)
        cubes[i,1] = np.ceil(d/n_div)-(np.floor((d-1)/n_div**2)*n_div)
        cubes[i,2] = d - (cubes[i,1]-1)*n_div - (cubes[i,0]-1)*n_div**2

    rs0 = (2*cubes-1)/n_div - 1

    return rs0


def generate_constraints(rs0):

    lbr0=rs0 - 4./25.  
    ubr0=rs0 + 4./25.  

    lbr0[lbr0<-1] = -1.01
    ubr0[ubr0>1] = 1.01

    return lbr0, ubr0
