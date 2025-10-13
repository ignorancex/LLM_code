import os
from dataclasses import dataclass

import gymnasium as gym
import h5py
import numpy as np
import pyrallis
import torch
from .cleanrl_ppo import Agent
from shimmy import DmControlCompatibilityV0
from tqdm.auto import trange

import src.dcs.suite as suite


@dataclass
class Args:
    checkpoint_path: str = "checkpoints"
    checkpoint_name: str = "100.pt"
    dcs_backgrounds_path: str = "DAVIS/JPEGImages/480p"
    dcs_backgrounds_split: str = "train"
    dcs_difficulty: str = "easy"
    dcs_dynamic: bool = True
    # 64px like in dreamer/td-mpc like methods
    dcs_img_hw: int = 64
    greedy_actions: bool = True
    num_trajectories: int = 1000
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


def make_env(args, config):
    def thunk():
        dm_env = suite.load(
            domain_name=config["domain"],
            task_name=config["task"],
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
        env = gym.wrappers.DtypeObservation(env, np.float32)
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


@pyrallis.wrap()
def main(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    config = torch.load(os.path.join(args.checkpoint_path, "config.pt"))
    checkpoint = torch.load(os.path.join(args.checkpoint_path, args.checkpoint_name), map_location=device)

    init_env = gym.vector.SyncVectorEnv([make_env(args, config) for i in range(1)])
    agent = Agent(init_env, hidden_dim=config["hidden_dim"]).to(device)
    agent.load_state_dict(checkpoint)

    dataset_returns = []
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with h5py.File(args.save_path, "x", track_order=True) as df:
        df.attrs["domain_name"] = config["domain"]
        df.attrs["task_name"] = config["task"]
        df.attrs["difficulty"] = args.dcs_difficulty
        df.attrs["dynamic"] = args.dcs_dynamic
        df.attrs["img_hw"] = args.dcs_img_hw
        df.attrs["split"] = args.dcs_backgrounds_split

        for idx in trange(args.num_trajectories):
            traj_return = 0.0
            pixels = []
            actions = []
            states = []

            env = make_env(args, config)()
            obs, info = env.reset(seed=args.seed + idx)
            done = False
            while not done:
                with torch.no_grad():
                    action = agent.get_action_and_value(
                        torch.tensor(obs[None], device=device), greedy=args.greedy_actions
                    )[0].cpu()
                    action = np.asarray(action.squeeze())

                # recording obs and corresponding action
                pixels.append(info["dcs_pixels"])
                states.append(obs)
                actions.append(action)
                # stepping in the env
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                traj_return += reward

            # writing to the dataset
            group = df.create_group(str(idx))
            group.create_dataset("obs", shape=(len(pixels), *pixels[0].shape), data=np.array(pixels), dtype=np.uint8)
            group.create_dataset(
                "states", shape=(len(states), *states[0].shape), data=np.array(states), dtype=np.float32
            )
            group.create_dataset(
                "actions", shape=(len(actions), *actions[0].shape), data=np.array(actions), dtype=np.float32
            )
            group.attrs["traj_return"] = traj_return
            dataset_returns.append(traj_return)
            print(f"Collected trajectory {idx}.")

        df.attrs["dataset_return"] = np.mean(dataset_returns)

    print("Done! Mean dataset return: ", np.mean(dataset_returns))


if __name__ == "__main__":
    main()
