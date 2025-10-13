import os
import time
from dataclasses import dataclass

import gymnasium as gym
import h5py
import numpy as np
import pyrallis
import torch
from shimmy import DmControlCompatibilityV0
from stable_baselines3 import SAC
from tqdm.auto import tqdm, trange

import src.dcs.suite as suite


@dataclass
class Args:
    checkpoint_path: str = "checkpoints/sb3-sac-checkpoint"
    dcs_backgrounds_path: str = "DAVIS/JPEGImages/480p"
    dcs_backgrounds_split: str = "train"
    dcs_difficulty: str = "easy"
    dcs_dynamic: bool = True
    # 64px like in dreamer/td-mpc like methods
    dcs_img_hw: int = 64
    greedy_actions: bool = True
    num_trajectories: int = 1000
    num_vec_envs: int = 10
    save_path: str = "data.hdf5"
    seed: int = 0
    cuda: bool = True


class PixelsToInfo(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({k: v for k, v in env.observation_space.items() if k != "pixels"})

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        pixels = obs.pop("pixels")
        info["dcs_pixels"] = pixels
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        pixels = obs.pop("pixels")
        info["dcs_pixels"] = pixels
        return obs, reward, terminated, truncated, info


def make_env(args):
    def thunk():
        dm_env = suite.load(
            domain_name="humanoid",
            task_name="walk",
            difficulty=args.dcs_difficulty,
            dynamic=args.dcs_dynamic,
            background_dataset_path=args.dcs_backgrounds_path,
            background_dataset_videos=args.dcs_backgrounds_split,
            pixels_only=False,
            render_kwargs=dict(height=args.dcs_img_hw, width=args.dcs_img_hw),
        )
        env = DmControlCompatibilityV0(dm_env, render_mode="rgb_array")
        env = PixelsToInfo(env)
        env = gym.wrappers.FlattenObservation(env)
        return env

    return thunk


@pyrallis.wrap()
def main(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    agent = SAC.load(args.checkpoint_path, device=device)

    dataset_returns = []
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with h5py.File(args.save_path, "x", track_order=True) as df:
        df.attrs["domain_name"] = "humanoid"
        df.attrs["task_name"] = "walk"
        df.attrs["difficulty"] = args.dcs_difficulty
        df.attrs["dynamic"] = args.dcs_dynamic
        df.attrs["img_hw"] = args.dcs_img_hw
        df.attrs["split"] = args.dcs_backgrounds_split

        group_idx = 0
        assert args.num_trajectories % args.num_vec_envs == 0
        print("Init vec env...")
        env = gym.vector.AsyncVectorEnv([make_env(args) for _ in range(args.num_vec_envs)])
        print("Done, start collecting...")

        for idx in trange(args.num_trajectories // args.num_vec_envs):
            s_time = time.time()
            tqdm.write(f"Start seed: {args.seed + idx * args.num_vec_envs}")
            obs, info = env.reset(seed=args.seed + idx * args.num_vec_envs)
            # TODO: better to pre-allocate numpy arrays
            pixels = []
            actions = []
            states = []
            rewards = []

            # WARN: true only for DCS (as all envs have 1k steps)!!!
            for step in range(1000):
                with torch.no_grad():
                    action, _ = agent.predict(obs, deterministic=args.greedy_actions)

                # recording obs and corresponding action
                pixels.append(info["dcs_pixels"])
                states.append(obs)
                actions.append(action)
                # stepping in the env
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)

            pixels = np.array(pixels).swapaxes(0, 1)
            actions = np.array(actions).swapaxes(0, 1)
            states = np.array(states).swapaxes(0, 1)
            rewards = np.array(rewards).swapaxes(0, 1)

            # print(pixels.shape, actions.shape, states.shape, rewards.shape)
            # writing to the dataset
            for i in range(args.num_vec_envs):
                group = df.create_group(str(group_idx))
                group.create_dataset("obs", shape=(1000, *pixels[i][0].shape), data=pixels[i], dtype=np.uint8)
                group.create_dataset("states", shape=(1000, *states[i][0].shape), data=states[i], dtype=np.float32)
                group.create_dataset("actions", shape=(1000, *actions[i][0].shape), data=actions[i], dtype=np.float32)
                group.attrs["traj_return"] = rewards[i].sum()
                dataset_returns.append(rewards[i].sum())
                group_idx += 1

            tqdm.write(
                f"Collected {args.num_vec_envs} trajectories in {time.time() - s_time}s. Total: {args.num_vec_envs * idx}\n"
            )

        df.attrs["dataset_return"] = np.mean(dataset_returns)

    print("Done! Mean dataset return: ", np.mean(dataset_returns))


if __name__ == "__main__":
    from multiprocessing import set_start_method

    set_start_method("spawn")
    main()
