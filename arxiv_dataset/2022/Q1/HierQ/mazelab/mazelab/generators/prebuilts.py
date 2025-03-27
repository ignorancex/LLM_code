import numpy as np

from .n_rooms_square import n_rooms_square
from .gridworld import gridworld, contracted_gridworld


def four_rooms() -> np.ndarray:
    # Canonical Four-Rooms environment from Sutton's Between MDPs and Semi-MDPS, 1999.
    x = n_rooms_square(4, 5, 1)

    x[7,7:-1] = x[6,7:-1]
    x[6,7:-1] = 0

    return x.astype(np.uint8)


def four_rooms_HierQ() -> np.ndarray:
    # Four rooms environment designed by Andrew Levy, 2019.
    x = n_rooms_square(4, 5, 2)

    x[:6, 6] = 1  # Block upper-right upper-left doorway.
    x[6, 10], x[6, 8] = x[6, 8], x[6, 10]  # Offset even gap to room-center
    x[10, 6], x[8, 6] = x[8, 6], x[10, 6]  # Offset even gap to room-center

    return x.astype(np.uint8)


def open_world() -> np.ndarray:
    # Open grid with soft corners as used by van Hasselt et al., 2021 https://arxiv.org/pdf/2007.01839.pdf.
    return contracted_gridworld(height=11, width=11, contract=3).astype(np.uint8)


def dyna_example() -> np.ndarray:
    # Prebuilt maze as illustrated in the dynaQ example in Chapter 8.2 of Sutton and Barto 2018.
    # Generate a 6x9 empty maze with padded borders
    x = np.zeros((6 + 2, 9 + 2), dtype=np.uint8)
    x[0, :] = x[-1, :] = x[:, 0] = x[:, -1] = 1
    
    # Add objects 
    x[[2, 3, 4], 3] = x[5, 6] = x[[1, 2, 3], -3] = 1
    
    return x.astype(np.uint8)


def fixed_maze(large: bool = False) -> np.ndarray:  # Maze macro function to get the same maze everytime.
    if large:
        return np.asarray([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                           [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                           [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                           [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                           [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                           [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                           [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                           [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                           [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                           [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                           [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                           [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                           [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                           [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                           [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                           [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    else:
        return np.asarray([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                           [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                           [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                           [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                           [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                           [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                           [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                           [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                           [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
