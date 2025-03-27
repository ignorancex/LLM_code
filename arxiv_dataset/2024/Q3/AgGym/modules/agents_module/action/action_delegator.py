import importlib
import gym
from gymnasium.spaces import MultiDiscrete

def init(self):
    if len(self.action_dim) > 0:
        self.action_space = gym.spaces.Discrete(len(self.action_dim))
        
        # self.action_space = MultiDiscrete([4]*71) 
        # self.action_space = gym.spaces.MultiDiscrete([4] * 25)
        self.pesticide_actions = {}
        for i in range(len(self.action_dim)):
            self.pesticide_actions[i] = self.action_dim[i]
        print(self.pesticide_actions)
    else:
        # continuous space wip
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
