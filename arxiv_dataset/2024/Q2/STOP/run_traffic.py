import numpy as np
#import matplotlib.pyplot as plt
import gym

import os
import sys
import shutil
import pdb
import argparse
import random
import copy

import torch

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from algos.callbacks.ContinualEvalCallback import ContinualEvalCallback

from policies import LQ, LSR, LCQ, LSCQ, StablebaselinePolicy, MaxWeight, Random, LASQ, LSQ

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumo.sumo_rl.environment.env import SumoEnvironment
from sumo.sumo_rl.environment.observations import UnboundedObservation
#from gridworld import Gridworld, GridAxis
import utils

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
# saving
parser.add_argument('--outfile', default = None)

# common setup
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--exp_name', type = str, required = True)
parser.add_argument('--env_name', type = str, required = True)
parser.add_argument('--algo_name', type = str, default = 'all')
parser.add_argument('--mdp_num', default = 0, type = int)
parser.add_argument('--r_mix_ratio', default = 1.0, type = float)
parser.add_argument('--opt_warmup_time', default = 1e6, type = float)
parser.add_argument('--opt_beta', default = 4e-6, type = float)
parser.add_argument('--gamma', default = 0.999, type = float)
parser.add_argument('--lr', default = 3e-4, type = float)
parser.add_argument('--moving_avg', default = 0.1, type = float)
parser.add_argument('--nu_bias', default = 0.1, type = float)
parser.add_argument('--pretrained_name', default = None, type = str)
parser.add_argument('--act_function', default = 'relu', type = str)
parser.add_argument('--reward_function', default = 'avg-q-len', type = str)
#parser.add_argument('--reward_transformation', default = 'id', type = str)
parser.add_argument('--state_transformation', default = 'id', type = str)
parser.add_argument('--normalize_rewards', default = False, type = str2bool)
parser.add_argument('--state_bound', default = np.inf, type = float)
parser.add_argument('--normalize_env', default = False, type = str2bool)
parser.add_argument('--anneal_lr', default = False, type = str2bool)

parser.add_argument('--batch_size', default = 64, type = int)
parser.add_argument('--replay_epochs', default = 10, type = int)

parser.add_argument('--truncated_horizon', default = 2048, type = int) # same as train_freq in DQN
parser.add_argument('--training_horizon', default = 10000, type = int)
parser.add_argument('--train_interaction_steps', default = 200000, type = int)
parser.add_argument('--deployed_interaction_steps', default = 250000, type = int)
parser.add_argument('--deployed_interaction_step_skip', default = 10, type = int)
parser.add_argument('--train_and_test', default = False, type = str2bool)

# DQN
parser.add_argument('--learning_starts', default = 0, type = int)
parser.add_argument('--exploration_fraction', default = 0.1, type = float)

FLAGS = parser.parse_args()

assert (FLAGS.deployed_interaction_steps % FLAGS.deployed_interaction_step_skip == 0)

log_dir = 'temp_{}'.format(FLAGS.outfile)
os.makedirs(log_dir, exist_ok=True)

def get_env():
    if FLAGS.env_name == 'traffic':
        if FLAGS.mdp_num == 0:
            net_file = 'sumo/nets/big-intersection/big-intersection.net.xml'
            route_file = 'sumo/nets/big-intersection/routes.rou.xml'
        elif FLAGS.mdp_num == 1:
            net_file = 'sumo/nets/big-intersection/big-intersection.net.xml'
            route_file = 'sumo/nets/big-intersection/routes_high.rou.xml'
        elif FLAGS.mdp_num == 2:
            net_file = 'sumo/nets/big-intersection/big-intersection.net.xml'
            route_file = 'sumo/nets/big-intersection/routes_vhigh.rou.xml'

    env = SumoEnvironment(
        net_file = net_file,
        route_file = route_file,
        single_agent=True,
        use_gui=False,
        observation_class = UnboundedObservation,
        state_trans = FLAGS.state_transformation,
        reward_fn = FLAGS.reward_function,
        begin_time = 0,
        num_seconds = FLAGS.deployed_interaction_steps,
        opt_beta = FLAGS.opt_beta,
        opt_warmup_time = FLAGS.opt_warmup_time
    )

    #env = Monitor(env, log_dir)
    return env

def _train_RL(env, algo_name, variant = None, state_transformation = None, fname = None):

    print ('training {} {}'.format(algo_name, variant))
    
    # if FLAGS.train_and_test:
    #     if FLAGS.normalize_env:
    #         env.env_method('set_horizon', FLAGS.training_horizon)
    #     else:
    #         env.set_horizon(FLAGS.training_horizon)
    # else:
    #     if FLAGS.normalize_env:
    #         env.env_method('set_horizon', -1)
    #     else:
    #         env.set_horizon(-1)
    
    pt_path = None
    if FLAGS.pretrained_name:
        pt_path = 'pretrained_policies/' + str(FLAGS.pretrained_name) + '_' + str(2)
        #pt_path = 'pretrained_policies/' + str(FLAGS.pretrained_name) + '_' + str(FLAGS.mdp_num)

    if FLAGS.normalize_env:
        policy_kwargs = dict(activation_fn = torch.nn.Tanh, action_mask_extractor = env.get_attr('mask_extractor')[0])
    else:
        policy_kwargs = dict(activation_fn = torch.nn.Tanh,\
                    action_mask_extractor = None)

    policy_type = 'MlpPolicy'

    if 'PPO' in algo_name or 'TRPO' in algo_name:
        if 'PPO' in algo_name:
            name = 'AR-PPO'
        elif 'TRPO' in algo_name:
            name = 'AR-TRPO'
        policy_kwargs['net_arch'] = [dict(pi=[64, 64], vf=[64, 64])]
        if 'Multi-VF' in algo_name:
            policy_kwargs['vf_weights'] = [1, 0] # stability + optimality
            policy_type = 'MlpMultiCriticPolicy'
    elif 'DQN' in algo_name:
        if 'QR' in algo_name:
            name = 'QR-DQN'
        else:
            name = 'DQN'
            policy_kwargs['net_arch'] = [64, 64]
            #policy_kwargs['boltzmann_exp'] = True
            #policy_kwargs['boltzmann_exp_temp'] = 0.1

    pi = StablebaselinePolicy(policy_type, name, env,
        gamma = FLAGS.gamma,
        learning_rate = FLAGS.lr,
        moving_avg = FLAGS.moving_avg,
        nu_bias = FLAGS.nu_bias,
        variant = variant,
        pretrained_path = pt_path,
        use_lcq = False,
        truncated_horizon = FLAGS.truncated_horizon,
        batch_size = FLAGS.batch_size,
        replay_epochs = FLAGS.replay_epochs,
        normalize_rewards = FLAGS.normalize_rewards,
        learning_starts = FLAGS.learning_starts,
        exploration_fraction = FLAGS.exploration_fraction,
        policy_kwargs = policy_kwargs,
        behavior_policy = None,#LCQ(env), # TODO not correct for continual learning since the policy getting trained should be acting in env not the beh
        augment_data = False,
        anneal_lr = FLAGS.anneal_lr) # NOTE: LCQ/MW will use logged queue lengths during decision making (but OK since monotonic)
    #callback = EvalCallback(env, best_model_save_path = None, log_path = log_dir, eval_freq = 2000,
    #            deterministic = True, render = False, verbose = 0, n_eval_episodes = 1)

    if FLAGS.train_and_test:
        # train and test
        callback = None
        total_timesteps = FLAGS.train_interaction_steps
    else:
        # continuing task
        total_timesteps = FLAGS.deployed_interaction_steps / env.delta_time
        callback = ContinualEvalCallback(log_freq = FLAGS.deployed_interaction_step_skip,\
                                        state_transformation = state_transformation,
                                        fname = fname,
                                        plot = False,
                                        plot_freq = 10000 / env.delta_time)

    pi.learn(total_timesteps = total_timesteps, callback = callback)
    stats = callback.get_stats() if callback else None
    return pi, stats

def run_experiment_algo(env, algo_name):

    fname = 'temp' if FLAGS.outfile is None else FLAGS.outfile
    print ('executing {} policy'.format(algo_name))

    if 'PPO' in algo_name\
        or 'TRPO' in algo_name or 'DQN' in algo_name:
        variant = None
        if 'PPO' in algo_name or 'TRPO-AR' in algo_name:
            variant = 'zhang'
        # elif 'PPO-D' in algo_name or 'TRPO-D' in algo_name:
        #     variant = 'discount'
        rl_res = []
        print ('executing {} {}'.format(algo_name, FLAGS.state_transformation))
        #env.set_state_transformation(FLAGS.state_transformation)
        #env.env_method('set_state_transformation', FLAGS.state_transformation)
        pi, pi_stats = _train_RL(env, algo_name = algo_name, state_transformation = FLAGS.state_transformation, fname = fname, variant = variant)
        if FLAGS.train_and_test:
            env.reset()
            if FLAGS.normalize_env:
                env.env_method('set_state_bound', np.inf)
                env.env_method('set_horizon', -1)
            else:
                env.set_state_bound(np.inf)
                env.set_horizon(-1)
            #env.env_method('set_horizon', -1)
            paths, _ = utils.collect_data(env, pi, 1, FLAGS.deployed_interaction_steps)
            rl_bl = paths[0]['avg_backlog']
            rl_bl = rl_bl[:FLAGS.deployed_interaction_steps:FLAGS.deployed_interaction_step_skip]
        else:
            rl_bl = pi_stats['backlog']
            rl_bl = rl_bl[:int(FLAGS.deployed_interaction_steps / FLAGS.deployed_interaction_step_skip)] # remove extra fluff that sb3 has
        rl_res.append(rl_bl)
        #utils.plot_heatmap(pi, '{}_{}'.format(fname, 'zhang-' + t), transformation = t)
        backlogs = rl_res
    elif algo_name == 'LCQ' or algo_name == 'MW' or algo_name == 'LSCQ' or algo_name == 'Rand':    
        env.set_horizon(-1)
        if algo_name == 'LCQ':
            pi = LCQ(env)
        elif algo_name == 'MW':
            pi = MaxWeight(env)
        elif algo_name == 'LSCQ':
            pi = LSCQ(env)
        elif algo_name == 'Rand':
            pi = Random(env, use_connections = True)
        paths, _ = utils.collect_data(env, pi, 1, FLAGS.deployed_interaction_steps)
        bl = paths[0]['avg_backlog']
        bl = bl[:FLAGS.deployed_interaction_steps:FLAGS.deployed_interaction_step_skip]
        backlogs = [bl]
    return backlogs

def main():
    seed = FLAGS.seed
    utils.set_seed_everywhere(seed)
    env = get_env()
    
    denom = np.arange(1, FLAGS.deployed_interaction_steps / FLAGS.deployed_interaction_step_skip + 1)

    # (10 * (250 / 50) / 250) * 1000
    per_step_grad_updates = FLAGS.replay_epochs * (FLAGS.truncated_horizon / FLAGS.batch_size)
    replay_ratio = per_step_grad_updates / FLAGS.truncated_horizon
    #total_gradient_updates = FLAGS.interaction_steps * replay_ratio
    print (replay_ratio)
    #print (total_gradient_updates)
    #algos = ['PPO-M', 'PPO-Z', 'LCQ', 'MW', 'LSCQ', 'Rand-CF', 'Rand-CT']
    #algos = ['PPO-Z-SL', 'PPO-Z-MS', 'PPO-Z-MSSL', 'LCQ', 'MW', 'LSCQ', 'Rand-CF', 'Rand-CT']
    #algos = ['PPO-Z-SL', 'PPO-Z', 'LCQ', 'MW', 'LSCQ']

    algo_name = FLAGS.algo_name
    if algo_name == 'all':
        algos = ['PPO-Z-SL-{}-{}'.format(replay_ratio, FLAGS.truncated_horizon), 'LCQ', 'MW']
        backlogs = run_experiment_stoch(env)
    else:
        algos = [algo_name]
        backlogs = run_experiment_algo(env, algo_name)
   
    avg_backlogs = [np.divide(np.cumsum(backlog), denom) for backlog in backlogs]
    #avg_backlogs = [backlog for backlog in backlogs]

    summary = {
        'results': {},
        'seed': seed,
        'hp': {
            'truncated_horizon': FLAGS.truncated_horizon,
            'training_horizon': FLAGS.training_horizon,
            'lr': FLAGS.lr,
            'gamma': FLAGS.gamma,
            'replay_epochs': FLAGS.replay_epochs,
            'batch_size': FLAGS.batch_size,
            'r_mix': FLAGS.r_mix_ratio,
            'replay_ratio': replay_ratio,
            'opt_beta': FLAGS.opt_beta,
            'opt_warmup_time': FLAGS.opt_warmup_time,
            'anneal_lr': FLAGS.anneal_lr
            #'total_grad_updates': total_gradient_updates
        },
    }

    for idx, algo in enumerate(algos):
        summary['results'][algo] = {
            'avg_backlog': avg_backlogs[idx],
            #'avg_backlog_changes': backlog_changes[idx]
        }
    print (summary)
    #x = summary['results']['PPO-Z-SL']['avg_backlog']
    np.save(FLAGS.outfile, summary)
    
    #shutil.move('temp_{}'.format(FLAGS.outfile) + '/best_model.zip', FLAGS.outfile + '_best_model.zip')
    #shutil.move('temp_{}'.format(FLAGS.outfile) + '/evaluations.npz', FLAGS.outfile + '_evaluations.npz')

if __name__ == '__main__':
    main()

         

