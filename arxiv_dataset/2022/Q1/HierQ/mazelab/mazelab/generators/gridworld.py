import numpy as np


def gridworld(height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width))
    grid[:, 0] = grid[:, -1] = grid[0, :] = grid[-1, :] = 1
    
    return grid.astype(np.uint8)


def contracted_gridworld(height: int, width: int, contract: int) -> np.ndarray:
    grid = gridworld(height=height, width=width)

    n = contract + 1
    for i in range(n):
        grid[-(n - i):, i + 1] = grid[:(i + 1), -(n - i + 1):] = 1

    return grid.astype(np.uint8)
