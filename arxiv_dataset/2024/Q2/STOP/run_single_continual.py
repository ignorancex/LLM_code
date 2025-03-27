import numpy as np
#import matplotlib.pyplot as plt

import os
import shutil
import pdb
import argparse
import random
import copy

import torch

from policies import LQ, LSR, LCQ, LSCQ, MaxWeight, Random,\
LASQ, LSQ, Threshold, LSQNModel, MWNModel, LQNModel, CleanRLPolicy
from server_allocation import SAQueue, SANetwork
from nmodel import NModelNetwork
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
parser.add_argument('--gamma', default = 0.999, type = float)
parser.add_argument('--lr', default = 3e-4, type = float)
parser.add_argument('--act_function', default = 'relu', type = str)
parser.add_argument('--reward_function', default = 'opt', type = str)
parser.add_argument('--state_transformation', default = 'id', type = str)
parser.add_argument('--state_bound', default = np.inf, type = float)
parser.add_argument('--use_action_mask', default = False, type = str2bool)
parser.add_argument('--lyp_power', default = 1., type = float)

parser.add_argument('--replay_epochs', default = 10, type = int)
parser.add_argument('--adam_beta', default = 0.9, type = float)

parser.add_argument('--truncated_horizon', default = 2048, type = int) # same as train_freq in DQN
parser.add_argument('--deployed_interaction_steps', default = 250000, type = int)
parser.add_argument('--deployed_interaction_step_skip', default = 10, type = int)

FLAGS = parser.parse_args()

assert (FLAGS.deployed_interaction_steps % FLAGS.deployed_interaction_step_skip == 0)

log_dir = 'temp_{}'.format(FLAGS.outfile)
os.makedirs(log_dir, exist_ok=True)

def get_env():
    if FLAGS.env_name == 'queue':
        if FLAGS.mdp_num == 0:
            # fully connected case
            q1_info = {
                'arrival': {
                    'is_stationary': True,
                    'prob': 0.2
                },
                'service': {
                    'is_stationary': True,
                    'prob': 0.3
                },
                'connection': {
                    'is_stationary': True,
                    'prob': 1.,
                }
            }
            q2_info = {
                'arrival': {
                    'is_stationary': True,
                    'prob': 0.1
                },
                'service': {
                    'is_stationary': True,
                    'prob': 0.8
                },
                'connection': {
                    'is_stationary': True,
                    'prob': 1.
                }
            }
            q1 = SAQueue('0', q1_info)
            q2 = SAQueue('1', q2_info)
            qs = np.array([q1, q2])
        elif FLAGS.mdp_num == 1:
            # Figure 4: https://proceedings.allerton.csl.illinois.edu/media/files/0062.pdf
            q1_info = {
                'arrival': {
                    'is_stationary': True,
                    'prob': 0.2
                },
                'service': {
                    'is_stationary': True,
                    'prob': 0.3
                },
                'connection': {
                    'is_stationary': True,
                    'prob': 0.95,
                }
            }
            q2_info = {
                'arrival': {
                    'is_stationary': True,
                    'prob': 0.1
                },
                'service': {
                    'is_stationary': True,
                    'prob': 0.8
                },
                'connection': {
                    'is_stationary': True,
                    'prob': 0.5
                }
            }
            q1 = SAQueue('0', q1_info)
            q2 = SAQueue('1', q2_info)
            qs = np.array([q1, q2])
        elif FLAGS.mdp_num == 2:
            # harder version of the above, lower connectivity to queue 1
            q1_info = {
                'arrival': {
                    'is_stationary': True,
                    'prob': 0.2
                },
                'service': {
                    'is_stationary': True,
                    'prob': 0.3
                },
                'connection': {
                    'is_stationary': True,
                    'prob': 0.7,
                }
            }
            q2_info = {
                'arrival': {
                    'is_stationary': True,
                    'prob': 0.1
                },
                'service': {
                    'is_stationary': True,
                    'prob': 0.8
                },
                'connection': {
                    'is_stationary': True,
                    'prob': 0.5
                }
            }
            q1 = SAQueue('0', q1_info)
            q2 = SAQueue('1', q2_info)
            qs = np.array([q1, q2])
        elif FLAGS.mdp_num == 3:
            configs = [(0.05, 0.9), (0.01, 0.85), (0.2, 0.95), (0.4, 0.75),\
                    (0.05, 0.9), (0.01, 0.9), (0.02, 0.85), (0.01, 0.9),\
                    (0.015, 0.9), (0.01, 0.85)]
            qs = []
            for idx, con in enumerate(configs):
                info = {
                    'arrival': {
                        'is_stationary': True,
                        'prob': con[0]
                    },
                    'service': {
                        'is_stationary': True,
                        'prob': con[1]
                    },
                    'connection': {
                        'is_stationary': True,
                        'prob': 1.,
                    }
                }
                q = SAQueue(str(idx), info)
                qs.append(q)
            qs = np.array(qs)
    elif FLAGS.env_name == 'nmodel':
        if FLAGS.mdp_num == 0:
            load = 0.99
            arrivals = [1.3 * load, 0.4 * load]
            holding_costs = [3., 1.]
            mus = [1., 1. / 2, 1.]
        elif FLAGS.mdp_num == 1:
            load = 0.99
            arrivals = [1.3 * load, 0.4 * load]
            holding_costs = [3., 1.]
            mus = [1., 1. / 2, 1.]
        elif FLAGS.mdp_num == 2:
            load = 0.95
            arrivals = [0.9, 0.8]
            holding_costs = [3., 1.]
            mus = [1., 0.9, 0.8]

    if FLAGS.env_name == 'gridworld' or FLAGS.env_name == 'queue':
        env = SANetwork(qs, reward_func = FLAGS.reward_function,\
                state_trans = FLAGS.state_transformation,
                gridworld = FLAGS.env_name == 'gridworld',
                state_bound = FLAGS.state_bound,
                lyp_power = FLAGS.lyp_power)
        print ('stable policy exists {}'.format(env.is_stable()))
    elif FLAGS.env_name == 'nmodel':
        env = NModelNetwork(arrivals, 
                mus,
                holding_costs,
                reward_func = FLAGS.reward_function,\
                state_trans = FLAGS.state_transformation,
                state_bound = FLAGS.state_bound,
                lyp_power = FLAGS.lyp_power)
    return env

def _train_RL(env, algo_name, variant = None, state_transformation = None, fname = None):

    print ('training {} {}'.format(algo_name, variant))
    
    env.set_horizon(-1)
    
    pt_path = None

    env.set_use_mask(FLAGS.use_action_mask)

    pi = CleanRLPolicy(env,
        learning_rate = FLAGS.lr,
        gamma = FLAGS.gamma, 
        variant = variant,
        num_steps = FLAGS.truncated_horizon,
        update_epochs = FLAGS.replay_epochs,
        use_action_mask = FLAGS.use_action_mask,
        adam_beta = FLAGS.adam_beta
    )


    # continuing task
    total_timesteps = FLAGS.deployed_interaction_steps

    pi.learn(total_timesteps = total_timesteps)
    stats = pi.get_stats(FLAGS.deployed_interaction_step_skip)
    return pi, stats

def run_experiment_algo(env, algo_name):

    fname = 'temp' if FLAGS.outfile is None else FLAGS.outfile
    print ('executing {} policy'.format(algo_name))
    visited_native_states = None
    vis_ns = []
    pi_stats = None
    if algo_name == 'PPO' or 'STOP' in algo_name:
        variant = 'zhang'
        rl_res = []
        print ('executing {} {}'.format(algo_name, FLAGS.state_transformation))
        #env.set_state_transformation(FLAGS.state_transformation)
        #env.env_method('set_state_transformation', FLAGS.state_transformation)
        pi, pi_stats = _train_RL(env, algo_name = algo_name, state_transformation = FLAGS.state_transformation, fname = fname, variant = variant)
        rl_bl = pi_stats['backlog']
        rl_bl = rl_bl[:int(FLAGS.deployed_interaction_steps / FLAGS.deployed_interaction_step_skip)] # remove extra fluff that sb3 has
        visited_native_states = []#pi_stats['visited_native_states']
        vis_ns.append(visited_native_states)
        rl_res.append(rl_bl)
        #utils.plot_heatmap(pi, '{}_{}'.format(fname, 'zhang-' + t), transformation = t)
        backlogs = rl_res
    elif algo_name == 'LCQ'\
        or algo_name == 'MW'\
        or algo_name == 'LSCQ'\
        or algo_name == 'Rand'\
        or algo_name == 'Thresh'\
        or algo_name == 'MWN'\
        or algo_name == 'LSQN'\
        or algo_name == 'LQN':    
        env.set_horizon(-1)
        if algo_name == 'LCQ':
            pi = LCQ(env)
        elif algo_name == 'MW':
            pi = MaxWeight(env)
        elif algo_name == 'LSCQ':
            pi = LSCQ(env)
        elif algo_name == 'Rand':
            pi = Random(env, smart = True)
        elif algo_name == 'Thresh':
            pi = Threshold(env)
        elif algo_name == 'MWN':
            pi = MWNModel(env)
        elif algo_name == 'LSQN':
            pi = LSQNModel(env)
        elif algo_name == 'LQN':
            pi = LQNModel(env)
        paths, _ = utils.collect_data(env, pi, 1, FLAGS.deployed_interaction_steps)
        bl = paths[0]['avg_backlog']
        bl = bl[:FLAGS.deployed_interaction_steps:FLAGS.deployed_interaction_step_skip]
        backlogs = [bl]
    return backlogs, vis_ns, pi_stats

def main():
    seed = FLAGS.seed
    utils.set_seed_everywhere(seed)
    env = get_env()
    
    denom = np.arange(1, FLAGS.deployed_interaction_steps / FLAGS.deployed_interaction_step_skip + 1)

    algo_name = FLAGS.algo_name
    algos = [algo_name]
    backlogs, visited_native_states, pi_stats = run_experiment_algo(env, algo_name)
   
    avg_backlogs = [np.divide(np.cumsum(backlog), denom) for backlog in backlogs]
    #avg_backlogs = [backlog for backlog in backlogs]

    summary = {
        'results': {},
        'seed': seed,
        'hp': {
            'truncated_horizon': FLAGS.truncated_horizon,
            'lr': FLAGS.lr,
            'gamma': FLAGS.gamma,
            'replay_epochs': FLAGS.replay_epochs,
        },
    }
    for idx, algo in enumerate(algos):
        summary['results'][algo] = {
            'avg_backlog': avg_backlogs[idx],
            # 'unstable_adv_mean': pi_stats['unstable_adv_mean'] if pi_stats else 0,
            # 'unstable_frac': pi_stats['unstable_frac'] if pi_stats else 0
            #'visited_native_states': visited_native_states[idx]
            #'avg_backlog_changes': backlog_changes[idx]
        }
    print (summary)
    np.save(FLAGS.outfile, summary)

if __name__ == '__main__':
    main()

         

