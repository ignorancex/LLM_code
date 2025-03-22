import gymnasium as gym
import numpy as np

class RandomEnv(gym.Env):

    def __init__(self):
        gym.Env.__init__(self)

    # Methods to override in child envs:
    # ----------------------------
    def get_search_bounds_mean(self, index):
        """Get search space for current randomized parameter at index `index`"""
        raise NotImplementedError

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for current randomized parameter at index `index`"""
        return -np.inf

    def get_task_upper_bound(self, index):
        """Returns highest feasible value for current randomized parameter at index `index`"""
        return np.inf

    def get_task(self):
        """Get current dynamics parameters"""
        raise NotImplementedError

    def set_task(self, *task):
        """Set dynamics parameters to <task>"""
        raise NotImplementedError
    # ----------------------------

    def sample_task(self):
        """Sample random dynamics parameters uniformly"""
        return np.random.uniform(self.min_task, self.max_task, self.min_task.shape)

    def set_task_search_bounds(self):
        """Sets the parameter search bounds based on how they are specified in get_search_bounds_mean"""
        dim_task = len(self.get_task())
        for i in range(dim_task):
            b = self.get_search_bounds_mean(i)
            self.min_task[i], self.max_task[i] = b[0], b[1]

    def get_task_search_bounds(self):
        dim_task = len(self.get_task())
        min_task = np.empty(dim_task)
        max_task = np.empty(dim_task)
        for i in range(dim_task):
            b = self.get_search_bounds_mean(i)
            min_task[i], max_task[i] = b[0], b[1]
        return min_task, max_task

    def sample_tasks(self, num_tasks=1):
        return np.stack([self.sample_task() for _ in range(num_tasks)])

    def set_random_task(self):
        """Sample and set random parameters
            
            Optionally keeps track of the sampled
            random parameters.
        """
        task = np.random.uniform(self.min_task, self.max_task, self.min_task.shape)
        self.set_task(*task)
