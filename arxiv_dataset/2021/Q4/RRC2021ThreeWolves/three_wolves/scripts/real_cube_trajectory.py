#!/usr/bin/python3
import sys
import json
from three_wolves.envs import contact_cube_env
from stable_baselines3 import SAC

def main():
    # the goal is passed as JSON string
    try:
        goal_json = sys.argv[1]
        goal_trajectory = json.loads(goal_json)
    except IndexError:
        goal_trajectory = None
    env = contact_cube_env.RealContactControlEnv(goal_trajectory=goal_trajectory)
    log_filename = f"/userhome/position_model.zip"
    policy = SAC.load(log_filename)

    observation = env.reset()
    done = False
    while not done:
        action = policy.predict(observation)[0]
        observation, reward, done, info = env.step(action)
        t = info["time_index"]
        print(f"{t} reward:", reward)


if __name__ == "__main__":
    main()
