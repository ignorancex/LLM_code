from dm_control import suite
import numpy as np
import imageio.v3 as iio
import gymnasium as gym
from shimmy import DmControlCompatibilityV0
from stable_baselines3 import SAC


def make_env():
    def thunk():
        dm_env = suite.load(
            domain_name="humanoid",
            task_name="walk",
            task_kwargs=None,
            environment_kwargs=None,
            visualize_reward=False,
        )
        env = DmControlCompatibilityV0(dm_env, render_mode="rgb_array", render_kwargs=dict(height=128, width=128))
        env = gym.wrappers.FlattenObservation(env)
        return env

    return thunk


def main():
    env = make_env()()
    model = SAC("MlpPolicy", env, device="cuda", verbose=1)
    model.learn(total_timesteps=2_000_000)
    model.save("/home/jovyan/nikulin/lapo-sb3-checkpoints/sac-humanoid-run")

    # model = SAC.load("checkpoints/sac-humanoid-walk")
    # images = []
    # total_reward = 0.0
    # obs, info = env.reset()
    # done = False
    # images.append(env.render())
    # while not done:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     total_reward += reward
    #     done = terminated or truncated
    #     images.append(env.render())

    # print(np.array(images).shape)
    # iio.imwrite("rollout_pixels.mp4", np.array(images), format_hint=".mp4", fps=32, macro_block_size=1)
    # print("Total reward:", total_reward)


if __name__ == "__main__":
    main()
