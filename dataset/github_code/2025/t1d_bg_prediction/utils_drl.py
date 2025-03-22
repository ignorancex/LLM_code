import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.metrics import root_mean_squared_error

L_BOUND = 20.0
U_BOUND = 420.0


class DiatrendEnv(gym.Env):
    def __init__(self, df, input_cols, observation_size, action_size, seed):
        self.observation_space = spaces.Box(
            low=L_BOUND, high=U_BOUND, shape=(observation_size,)
        )
        self.action_space = spaces.Box(low=L_BOUND, high=U_BOUND, shape=(action_size,))

        self.seed = seed
        self.df = df
        self.input_cols = input_cols
        self.observation_size = observation_size
        self.action_size = action_size

        self.idx = 0
        self.t = 0
        self.true_state = np.zeros((len(self.input_cols),))
        self.pred_state = np.zeros((observation_size,))

    def _get_obs(self):
        if self.idx == len(self.df):
            self.idx = 0
            self.t = 0
            print("Restarted from idx 0")
            self.df = self.df.sample(frac=1, random_state=self.seed, ignore_index=True)
        self.true_state = self.df.iloc[self.idx][self.input_cols].values.astype(float)
        self.pred_state = self.true_state[self.t : self.t + self.observation_size]

    def _get_reward(self, a, b):
        error = root_mean_squared_error(a, b) / (U_BOUND - L_BOUND)
        return 1 - error

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._get_obs()
        return self.pred_state, {}

    def step(self, action):
        self.pred_state = np.concatenate((self.pred_state[self.action_size :], action))
        y_true = self.true_state[
            self.t
            + self.action_size : self.t
            + self.observation_size
            + self.action_size
        ]
        reward = self._get_reward(y_true, self.pred_state)

        self.t += 1
        if self.t == int(self.observation_size / self.action_size):
            terminated = True
            self.idx += 1
            self.t = 0
            self._get_obs()
        else:
            terminated = False

        return self.pred_state, reward, terminated, False, {}
