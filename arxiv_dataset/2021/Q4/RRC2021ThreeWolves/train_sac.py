from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.monitor import Monitor
import os
from three_wolves.envs.utilities import model_utils
from three_wolves.envs.phase_cube_env import PhaseControlEnv
from three_wolves.envs.position_cube_env import PositionControlEnv
from trifinger_simulation.tasks import move_cube_on_trajectory as mct
import numpy as np
import argparse


def get_arguments():
    parser = argparse.ArgumentParser("Tri-finger")
    # Environment
    parser.add_argument("--controller-type", '-c', type=str, choices=['position', 'force'],
                        help="type of model controller")
    parser.add_argument("--action-type", '-a', type=str, choices=['full', 'residuals', 'RefRsd'],
                        help="type of model action")
    parser.add_argument("--model-name", '-m', type=str, help="name of model(save path)")
    parser.add_argument("--run-mode", '-r', type=str, choices=['train', 'test'], help="train or test model")
    parser.add_argument("--device", '-d', type=int, default=0, help="cuda device")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    log_dir = f"three_wolves/deep_whole_body_controller/model_save/" \
              f"{args.controller_type}_{args.action_type}_{args.model_name}/"
    # env_class = PhaseControlEnv
    env_class = PositionControlEnv
    if args.run_mode == 'train':
        env = env_class(goal_trajectory=None,
                        visualization=False,
                        args=args)
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        callback = model_utils.SaveOnBestTrainingRewardCallback(check_freq=int(1e4), log_dir=log_dir)
        model = SAC(MlpPolicy, env, verbose=1, device=f'cuda:{args.device}')
        model.learn(total_timesteps=int(1e7), callback=callback, log_interval=int(1e4))
    else:
        env = env_class(goal_trajectory=None,
                        visualization=True,
                        args=args)
        model_utils.plot_results(log_dir)
        model = SAC.load(log_dir + "best_model.zip")
        SCORE = []
        obs = env.reset()
        for _ in range(10):
            for i in range(mct.EPISODE_LENGTH // 100):
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                print(info['eval_score'])
                if done:
                    obs = env.reset()
        # print(f'10 times Mean Score: {np.mean(SCORE)}')
