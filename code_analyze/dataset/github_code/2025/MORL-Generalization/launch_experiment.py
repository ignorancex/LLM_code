"""Launches an experiment on a given environment and algorithm.

Many parameters can be given in the command line, see the help for more infos.

Examples:
    python benchmark/launch_experiment.py --algo pcn --env-id deep-sea-treasure-v0 --num-timesteps 1000000 --gamma 0.99 --ref-point 0 -25 --wandb-entity openrlbenchmark --seed 0 --init-hyperparams "scaling_factor:np.array([1, 1, 1])"
"""

import argparse
import os
import subprocess
import copy
from distutils.util import strtobool

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import requests
from gymnasium.wrappers import FlattenObservation
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

from mo_utils.wrappers import Multi2SingleObjectiveWrapper
from mo_utils.evaluation import seed_everything
from mo_utils.experiments import (
    ALGOS,
    SINGLE_OBJECTIVE_ALGOS,
    ENVS_WITH_KNOWN_PARETO_FRONT,
    StoreDict,
)
from morl_generalization.utils import get_env_selection_algo_wrapper, MORecordVideo
from morl_generalization.generalization_evaluator import make_generalization_evaluator
from envs.register_envs import register_envs
from envs.mo_super_mario.utils import wrap_mario
from algos.single_policy.ser.mo_ppo import make_env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="Name of the algorithm to run", choices=ALGOS.keys(), required=True)
    parser.add_argument("--env-id", type=str, help="MO-Gymnasium id of the environment to run", required=True)
    parser.add_argument("--num-timesteps", type=int, help="Number of timesteps to train for", required=True)
    parser.add_argument("--gamma", type=float, help="Discount factor to apply to the environment and algorithm", required=True)
    parser.add_argument(
        "--ref-point", type=float, nargs="+", help="Reference point to use for the hypervolume calculation", required=True
    )
    parser.add_argument("--seed", type=int, help="Random seed to use", default=42)
    parser.add_argument("--log", type=lambda x: bool(strtobool(x)), help="Whether to enable wandb logging (default: True)", default=True)
    parser.add_argument("--wandb-entity", type=str, help="Wandb entity to use", required=False)
    parser.add_argument("--wandb-group", type=str, help="Wandb group to use for logging", required=False)
    parser.add_argument("--wandb-tags", type=str, nargs="+", help="Extra wandb tags for experiment versioning", required=False)
    parser.add_argument("--wandb-offline", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Whether to run wandb offline")
    parser.add_argument(
        "--record-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, the runs will be recorded with RecordVideo wrapper.",
    )
    parser.add_argument("--record-video-ep-freq", type=int, required=False, help="Record video frequency (in episodes).")
    parser.add_argument("--record-video-w-freq", type=int, required=False, help="Record video frequency (in number of weight evaluated).")
    parser.add_argument(
        "--init-hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Override hyperparameters to use for the initiation of the algorithm. Example: --init-hyperparams learning_rate:0.001 final_epsilon:0.1",
        default={},
    )

    parser.add_argument(
        "--train-hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Override hyperparameters to use for the train method algorithm. Example: --train-hyperparams num_eval_weights_for_front:10 timesteps_per_iter:10000",
        default={},
    )

    parser.add_argument(
        "--test-generalization",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Whether to test the generalizability of the algorithm (default: True)",
    )

    parser.add_argument(
        "--generalization-hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Override hyperparameters to use for the generalizability evaluation. \
            Example: --generalization-hyperparams num_eval_weights:100 num_eval_episodes:5",
        default={},
    )

    parser.add_argument(
        '--test-envs',
        type=str,
        default='',
        help='CSV string of test environments for evaluation during training.'
    )


    return parser.parse_args()


def parse_generalization_args(args):
    if args.test_generalization:
        assert "test_envs" != '', "test_envs must be provided if test_generalization is True"
        # assert args.record_video == False, "cannot record video when testing generalization because environments are vectorized"
        args.test_envs = args.test_envs.split(",")

        # default values if not provided
        args.generalization_hyperparams.setdefault("generalization_algo", "domain_randomization")
        args.generalization_hyperparams.setdefault("history_len", 1)
    
    return args

def make_envs(args):
    if args.record_video:
        assert sum(x is not None for x in [args.record_video_ep_freq, args.record_video_w_freq]) == 1, \
            "Must specify exactly one video recording trigger: record_video_ep_freq or record_video_w_freq"

    if "mario" in args.env_id.lower():
        env = mo_gym.make(args.env_id, death_as_penalty=True, time_as_penalty=True)
        eval_env = mo_gym.make(args.env_id, death_as_penalty=True, time_as_penalty=True, render_mode="rgb_array" if args.record_video else None)
    else:
        env = mo_gym.make(args.env_id)
        eval_env = mo_gym.make(args.env_id, render_mode="rgb_array" if args.record_video else None)

    env = MORecordEpisodeStatistics(env, gamma=args.gamma)

    if "highway" in args.env_id:
        env = FlattenObservation(env)
        eval_env = FlattenObservation(eval_env)
    elif "mario" in args.env_id.lower():
        env = wrap_mario(env)
        eval_env = wrap_mario(
            eval_env, 
            gym_id=args.env_id, 
            algo_name=args.algo,
            seed=args.seed,
            record_video=args.record_video, 
            record_video_ep_freq=args.record_video_ep_freq,
            record_video_w_freq=args.record_video_w_freq,
        )

    if args.algo in SINGLE_OBJECTIVE_ALGOS:
        print("Training single-objective agent... Converting multi-objective environment to single-objective environment")
        # no need to wrap eval_env because it is only used for evaluation because we want get vector reward for logging even
        # for single-objective algorithms
        env = Multi2SingleObjectiveWrapper(env)

    if args.test_generalization:
        env = get_env_selection_algo_wrapper(env, args.generalization_hyperparams)
        eval_env = get_env_selection_algo_wrapper(eval_env, args.generalization_hyperparams, is_eval_env=True)
        
        # allow for comprehensize evaluation of generalization
        eval_env = make_generalization_evaluator(eval_env, args)
    elif args.record_video and "mario" not in args.env_id.lower(): # wrap_mario already has record_video
        if args.record_video_ep_freq:
            eval_env = MORecordVideo(
                eval_env,
                video_folder=f"videos/{args.algo}/seed{args.seed}/{args.env_id}",
                episode_trigger=lambda ep: ep % args.record_video_ep_freq == 0,
                disable_logger=True
            )
        elif args.record_video_w_freq:
            eval_env = MORecordVideo(
                eval_env, 
                video_folder=f"videos/{args.algo}/seed{args.seed}/{args.env_id}/", 
                weight_trigger=lambda t: t % args.record_video_w_freq == 0,
                disable_logger=True
            )

    env.unwrapped.action_space.seed(args.seed)
    env.unwrapped.observation_space.seed(args.seed)
    eval_env.unwrapped.action_space.seed(args.seed)
    eval_env.unwrapped.observation_space.seed(args.seed)
    return env, eval_env

def main():
    register_envs()
    args = parse_args()
    args = parse_generalization_args(args)
    print(args)

    seed_everything(args.seed)

    if args.algo == "pgmorl":
        # PGMORL creates its own environments because it requires wrappers
        print(f"Instantiating {args.algo} on {args.env_id}")

        env_creator = make_env(args.env_id, seed=args.seed, idx=-1, run_name="PGMORL", gamma=args.gamma)
        eval_env = env_creator()
        temp_env = env_creator()
        if args.test_generalization:
            temp_env = get_env_selection_algo_wrapper(eval_env, args.generalization_hyperparams)
            eval_env = get_env_selection_algo_wrapper(eval_env, args.generalization_hyperparams, is_eval_env=True)

            eval_env = make_generalization_evaluator(eval_env, args)

        algo = ALGOS[args.algo](
            env=temp_env,
            env_id=args.env_id,
            origin=np.array(args.ref_point),
            gamma=args.gamma,
            log=args.log,
            seed=args.seed,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
            wandb_tags=args.wandb_tags,
            offline_mode=args.wandb_offline,
            generalization_hyperparams=args.generalization_hyperparams if args.test_generalization else None,
            **args.init_hyperparams,
        )
        print(algo.get_config())

        print("Training starts... Let's roll!")
        algo.train(
            total_timesteps=args.num_timesteps,
            eval_env=eval_env,
            ref_point=np.array(args.ref_point),
            known_pareto_front=None,
            test_generalization=args.test_generalization,
            **args.train_hyperparams,
        )

    else:
        env, eval_env = make_envs(args)
        
        print(f"Instantiating {args.algo} on {args.env_id}")
        if args.algo == "ols":
            args.init_hyperparams["experiment_name"] = "MultiPolicy MO Q-Learning (OLS)"
        elif args.algo == "gpi-ls":
            args.init_hyperparams["experiment_name"] = "MultiPolicy MO Q-Learning (GPI-LS)"

        algo = ALGOS[args.algo](
            env=env,
            gamma=args.gamma,
            log=args.log,
            seed=args.seed,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
            wandb_tags=args.wandb_tags,
            **args.init_hyperparams,
        )
        if args.env_id in ENVS_WITH_KNOWN_PARETO_FRONT:
            known_pareto_front = env.unwrapped.pareto_front(gamma=args.gamma)
        else:
            known_pareto_front = None

        print(algo.get_config())

        print("Training starts... Let's roll!")
        algo.train(
            total_timesteps=args.num_timesteps,
            eval_env=eval_env,
            ref_point=np.array(args.ref_point),
            known_pareto_front=known_pareto_front,
            test_generalization=args.test_generalization,
            **args.train_hyperparams,
        )


if __name__ == "__main__":
    main()
