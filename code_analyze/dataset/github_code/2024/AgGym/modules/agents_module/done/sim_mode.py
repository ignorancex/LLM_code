import numpy as np
from utils import env_utils

def main(self):
    # print(self.timestep, self.max_timestep, self.sim_mode)
    if self.sim_mode == 'growthseason':
        if np.array_equal(self.grid, self.terminal_grid) == True:
            done = True
            # print(f"before penalty {reward}")
            # print(f"after penalty {reward}")
            if self.mode == "eval" or (self.mode == "train" and self.episode % int(self.max_episode) == 0 and self.plot_progression == 'True'):
                # or (self.mode == "train" and self.episode % 1000 == 0)
                env_utils.make_gif(self)
            
        elif self.timestep == self.gs_end - 1:
            done = True
            # print(f"Before bonus {reward}")
            # print(f"After bonus {reward}")
            if self.mode == "eval" or (self.mode == "train" and self.episode % int(self.max_episode) == 0 and self.plot_progression == 'True'):
                env_utils.make_gif(self)
        elif self.timestep == self.max_timestep - 1:
            # print('second')
            done = True
        else:
            done = False
    elif self.sim_mode == 'survival':
        # print('here')
        if np.array_equal(self.grid, self.terminal_grid) == True:
            # print('first')
            done = True
            # print(f"before penalty {reward}")
            # print(f"after penalty {reward}")
            if self.mode == "eval" or (self.mode == "train" and self.episode % int(self.max_episode) == 0 and self.plot_progression == 'True'):
                # or (self.mode == "train" and self.episode % 1000 == 0)
                env_utils.make_gif(self)

        elif self.timestep == self.max_timestep - 1:
            # print('second')
            done = True
            
            if self.mode == "eval" or (self.mode == "train" and self.episode % int(self.max_episode) == 0 and self.plot_progression == 'True'):
                env_utils.make_gif(self)
        else:
            done = False
    else:
        assert 1 == 2, f"Invalid sim_mode {self.sim_mode}"
            
    return done