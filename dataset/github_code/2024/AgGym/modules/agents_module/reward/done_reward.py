import numpy as np

def main(self, severity=0):
    reward = 0

    if self.sim_mode == 'growthseason':

        if np.array_equal(self.grid, self.terminal_grid) == True:
            reward -= 1000

        elif self.timestep == self.gs_end - 1:
            if np.array_equal(self.grid, self.healthy_grid) == True:
                reward += 1000
        else:
            reward = 0
    elif self.sim_mode == 'survival':

        if np.array_equal(self.grid, self.terminal_grid) == True:
            reward -= 1000
        elif self.timestep == self.max_timestep:
            if np.array_equal(self.grid, self.healthy_grid) == True:
                reward += 1000
        else:
            reward = 0

    return reward

