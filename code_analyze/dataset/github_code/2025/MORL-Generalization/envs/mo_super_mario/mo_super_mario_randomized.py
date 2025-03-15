
from typing import Optional
import warnings

import gymnasium as gym
import numpy as np
from envs.mo_super_mario.utils.mario_random_stages import SuperMarioBrosRandomStagesEnv
from gymnasium.utils import EzPickle, seeding

# from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from nes_py.nes_env import SCREEN_SHAPE_24_BIT

import mo_gymnasium as mo_gym

class MOSuperMarioBrosDR(SuperMarioBrosRandomStagesEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the SuperMarioBro environment.

    See [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) for more information.

    ## Reward Space
    The reward is a 5-dimensional vector:
    - 0: How far Mario moved in the x position
    - 1: Time penalty for how much time has passed between two time steps
    - 2: -25 if Mario died, 0 otherwise
    - 3: +100 if Mario collected coins, else 0
    - 4: Points for killing an enemy

    ## Episode Termination
    The episode terminates when Mario dies or reaches the flag.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_mode="pixel",
        lost_levels=False,
        target=None,
        objectives=["x_pos", "time", "death", "coin", "enemy"],
        death_as_penalty=False,
        time_as_penalty=False,
        render_mode: Optional[str] = None,
        stages=None,
        **kwargs,
    ):
        EzPickle.__init__(self, rom_mode, lost_levels, target, objectives, death_as_penalty, render_mode)
        self.render_mode = render_mode
        super().__init__(rom_mode, stages=stages)

        self.objectives = set(objectives)
        self.death_as_penalty = death_as_penalty
        self.time_as_penalty = time_as_penalty
        if self.death_as_penalty:  # death is not a separate objective
            self.objectives.discard("death")
        if self.time_as_penalty:  # time is not a separate objective
            self.objectives.discard("time")
        self.reward_dim = len(self.objectives)

        low = np.empty(self.reward_dim, dtype=np.float32)
        high = np.empty(self.reward_dim, dtype=np.float32)
        obj_idx = 0
        if "x_pos" in self.objectives:
            low[obj_idx] = -np.inf
            high[obj_idx] = np.inf
            obj_idx += 1
        if "time" in self.objectives:
            low[obj_idx] = -np.inf
            high[obj_idx] = 0.0
            obj_idx += 1
        if "death" in self.objectives:
            low[obj_idx] = -25.0
            high[obj_idx] = 0.0
            obj_idx += 1
        if "coin" in self.objectives:
            low[obj_idx] = 0.0
            high[obj_idx] = 100.0
            obj_idx += 1
        if "enemy" in self.objectives:
            low[obj_idx] = 0.0
            high[obj_idx] = np.inf

        self.reward_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(len(self.objectives),),
        )

        # observation space for the environment is static across all instances
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=SCREEN_SHAPE_24_BIT, dtype=np.uint8)

        # action space is a bitmap of button press values for the 8 NES buttons
        self.action_space = gym.spaces.Discrete(256)

        self.single_stage = True
        self.done_when_dead = True

    def reset(self, seed=None, **kwargs):
        self._np_random, seed = seeding.np_random(seed)  # this is not used
        self.coin = 0
        self.x_pos = 0
        self.time = 0
        self.score = 0
        self.stage_bonus = 0
        self.lives = 2
        obs = super().reset()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "human":
            super().render(mode="human")
        elif self.render_mode == "rgb_array":
            return super().render(mode="rgb_array")

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.single_stage and info["flag_get"]:
            self.stage_bonus = 10000 # most likely impossible if steps < 2000
            done = True

        """ Construct Multi-Objective Reward"""
        # [x_pos, time, death, coin, enemy]
        vec_reward = np.zeros(self.reward_dim, dtype=np.float32)
        obj_idx = 0

        # 1. x position
        if "x_pos" in self.objectives:
            xpos_r = info["x_pos"] - self.x_pos
            self.x_pos = info["x_pos"]
            # resolve an issue where after death the x position resets
            if xpos_r < -5:
                xpos_r = 0
            vec_reward[obj_idx] = xpos_r
            obj_idx += 1

        # 2. time penalty
        time_r = info["time"] - self.time
        self.time = info["time"]
        # time is always decreasing
        if time_r > 0:
            time_r = 0.0
        if "time" in self.objectives:
            vec_reward[obj_idx] = time_r
            obj_idx += 1

        # 3. death
        if self.lives > info["life"]:
            death_r = -25.0
        else:
            death_r = 0.0
        if "death" in self.objectives:
            vec_reward[obj_idx] = death_r
            obj_idx += 1

        # 4. coin
        coin_r = 0.0
        if "coin" in self.objectives:
            coin_r = (info["coins"] - self.coin) * 100 
            self.coin = info["coins"]
            vec_reward[obj_idx] = coin_r
            obj_idx += 1

        # 5. enemy
        if "enemy" in self.objectives:
            enemy_r = info["score"] - self.score
            if coin_r > 0:
                # coins are worth 200 points
                # can knock a coin block and kill an enemy on top simultaneously
                enemy_r -= 2 * coin_r
                if enemy_r < 0:
                    enemy_r = max(0, enemy_r) # enemy reward cannot be negative
                    warnings.warn("Enemy reward is negative. Setting it to 0.")
            if done:
                enemy_r = 0 # flag_get
            self.score = info["score"]
            vec_reward[obj_idx] = enemy_r
            obj_idx += 1

        if self.death_as_penalty:
            vec_reward += death_r # add death penalty to all objectives
        if self.time_as_penalty:
            vec_reward += time_r # add time penalty to all objectives
        ############################################################################

        if self.done_when_dead:
            # when Mario loses life, changes the state to the terminal
            if self.lives > info["life"] and info["life"] > 0:
                done = True

        self.lives = info["life"]

        # scale the reward, especially the coin and enemy reward
        vec_reward *= self.reward_space.shape[0] / 100

        info["score"] = info["score"] + self.stage_bonus

        # original reward in gym_super_mario_bros does not include the coin and enemy reward
        info["original_scalar_reward"] = xpos_r + time_r + death_r + (coin_r + enemy_r) * 0.0

        if self.render_mode == "human":
            self.render()

        return obs, vec_reward, bool(done), False, info


if __name__ == "__main__":
    from gymnasium.envs.registration import register
    from mo_utils.evaluation import seed_everything
    from gymnasium.wrappers.transform_observation import GrayscaleObservation, ResizeObservation
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from envs.mo_super_mario.utils.joypad_space import JoypadSpace
    from envs.mo_super_mario.utils.mario_video_wrapper import RecordMarioVideo
    import matplotlib.pyplot as plt

    seed_everything(42)

    register(
        id="MOSuperMarioBrosDR",
        entry_point="envs.mo_super_mario.mo_super_mario_randomized:MOSuperMarioBrosDR",
    )
    env = gym.make(
        "MOSuperMarioBrosDR", 
        render_mode="rgb_array", 
        death_as_penalty=True,
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = RecordMarioVideo(
        env,
        f"videos/test_mario/", 
        episode_trigger=lambda t: t % 10 == 0,
        disable_logger=True
    )
    # env = MaxAndSkipEnv(env, 4)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env)
    # env = FrameStack(env, 4)
    env = mo_gym.LinearReward(env)

    terminated = False
    env.unwrapped.reset_random()
    env.reset()
    while True:
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        print(r, info["vector_reward"], terminated, info["time"])
        # plt.figure()
        # plt.imshow(env.render(), cmap='gray', vmin=0, vmax=255)
        # plt.show()
        if terminated:
            env.unwrapped.reset_random()
            env.reset()