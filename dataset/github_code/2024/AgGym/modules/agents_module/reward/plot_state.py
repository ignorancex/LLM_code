import numpy as np

EMPTY = 0.0
DEAD = 1.0
INFECTED = 2.0
ALIVE = 3.0

def main(self):
	pesticide_reward = {0: 0, 1: -2, 2: -3, 3: -10}
	infect_count = np.count_nonzero(self.grid == INFECTED)
	dead_count = np.count_nonzero(self.grid == DEAD)

	reward = ((-infect_count * 1.0) + (-dead_count * 2.0) + pesticide_reward[self.action])

	return reward
