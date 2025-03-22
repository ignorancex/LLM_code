import numpy as np
import pdb
EMPTY = 0.0

def main(self):
    # if timestep == : pdb.set_trace()
    return self.grid[np.where(self.grid != EMPTY)] / 3
