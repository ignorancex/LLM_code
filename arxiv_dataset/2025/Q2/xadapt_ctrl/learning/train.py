#!/usr/bin/env python3
import argparse
import math
import os
from learning.ppo import PPO
from learning.dagger import DAGGER
from learning.module import StateHistoryEncoder
import numpy as np
import torch
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy

import torch.nn as nn


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--trial", type=int, default=1,
                        help="Policy trial number")
    parser.add_argument("--iter", type=int, default=100,
                        help="PPO iter number")
    parser.add_argument("--phase", type=int, default=1,
                        help="Training phase: 1 for PPO, 2 for DAGGER")
    return parser

def setup_environments(cfg, args):
    # Create training environment
    # TODO replace with your own environment
    train_env = QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)

    # Set random seed
    configure_random_seed(args.seed, env=train_env)

    # Create evaluation environment
    old_num_envs = cfg["simulation"]["num_envs"]
    cfg["simulation"]["num_envs"] = 1
    # TODO replace with your own environment
    eval_env = QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    cfg["simulation"]["num_envs"] = old_num_envs

    return train_env, eval_env

def main():
    args = parser().parse_args()
    rsg_root = os.path.dirname(os.path.abspath(__file__))

    # Load configurations
    cfg = YAML().load(
        open(rsg_root + "/hyperparam.yaml", "r")
    )

    # Setup environments
    train_env, eval_env = setup_environments(cfg, args)

    # Setup logging
    if args.phase == 1:
        log_dir = rsg_root + "/saved"
    else:
        log_dir = rsg_root + "/saved" + f'/Phase2_for_PPO{args.trial}_{args.iter}'
    os.makedirs(log_dir, exist_ok=True)

    if args.phase == 1:
        # Phase 1: PPO Training

        latent_size = cfg['observation_space']['latent_size']

        model = PPO(
            tensorboard_log=log_dir,
            policy="MlpPolicy",
            policy_kwargs=dict(
                activation_fn=torch.nn.Tanh,
                net_arch=[128, 128, latent_size, dict(pi=[256, 256], vf=[512, 512])],
                log_std_init=-0.5,
            ),
            env=train_env,
            eval_env=eval_env,
            use_tanh_act=True,
            gae_lambda=0.95,
            gamma=0.99,
            n_steps=500,
            ent_coef=0.0,
            vf_coef=0.5,
            min_BC_coef=0.1,
            BC_alpha=0.999,
            max_grad_norm=0.5,
            batch_size=25000,
            clip_range=0.2,
            use_sde=False,
            env_cfg=cfg,
            verbose=1,
            seed=args.seed,
        )

        model.learn(total_timesteps=10*1e7)

    else:
        ## Phase 2: DAGGER Training

        # Load PPO policy trained in phase 1
        weight = rsg_root + \
            "/saved/PPO_{0}/Policy/iter_{1:05d}.pth".format(
                args.trial, args.iter)
        env_rms = rsg_root + \
            "/saved/PPO_{0}/RMS/iter_{1:05d}.npz".format(args.trial, args.iter)
        device = get_device("auto")
        saved_variables = torch.load(weight, map_location=device)
        
        # Create policy object
        policy = MlpPolicy(**saved_variables["data"])
        policy.action_net = torch.nn.Sequential(
            policy.action_net, torch.nn.Tanh())
        policy.load_state_dict(saved_variables["state_dict"], strict=False)
        policy.to(device)

        # Load RMS
        eval_env.load_rms(env_rms)
        train_env.load_rms(env_rms)

        # Setup DAGGER components
        history_len = cfg['observation_space']['history_len']
        state_obs_size = cfg['observation_space']['state_obs_size']
        act_size = cfg['observation_space']['act_size']
        env_obs_size = cfg['observation_space']['env_obs_size']
        latent_size = cfg['observation_space']['latent_size']

        env_latent_encoder = StateHistoryEncoder(nn.Tanh,state_obs_size+act_size,
                                                                history_len, latent_size)
        env_latent_encoder.to(device)

        model = DAGGER(tensorboard_log=log_dir,
                        policy=policy,
                        env_latent_encoder=env_latent_encoder,
                        history_len=history_len,
                        state_obs_size=state_obs_size,
                        env_obs_size=env_obs_size,
                        latent_size=latent_size,
                        act_size=act_size,
                        env=train_env,
                        eval_env=eval_env,
                        use_tanh_act=True,
                        gae_lambda=0.95,
                        gamma=0.99,
                        n_steps=250,
                        ent_coef=0.0,
                        vf_coef=0.5,
                        max_grad_norm=0.5,
                        batch_size=25000,
                        clip_range=0.2,
                        use_sde=False,
                        env_cfg=cfg,
                        verbose=1,
                        seed=args.seed)

        model.learn(total_timesteps=int(5 * 1e6), log_interval=(5, 10))

if __name__ == "__main__":
    main()









