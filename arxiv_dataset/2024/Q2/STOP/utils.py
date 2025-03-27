import numpy as np
import pdb
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
from itertools import chain, combinations
import random
#from sklearn.preprocessing import MinMaxScaler
#from sklearn import preprocessing

import warnings
warnings.filterwarnings("error")

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def collect_data(env, policy, num_trajectory, truncated_horizon):
    paths = []
    num_samples = 0
    total_reward = 0.0
    for i_trajectory in range(num_trajectory):
        path = {}
        path['obs'] = []
        path['nobs'] = []
        path['acts'] = []
        path['rews'] = []
        path['avg_backlog'] = []
        path['avg_backlog_change'] = []
        state, _ = env.reset()
        sasr = []
        for i_t in range(truncated_horizon):
            action = policy(state, i_t)
            next_state, reward, done, _, info = env.step(action)
            path['obs'].append(state)
            path['acts'].append(action)
            path['rews'].append(reward)
            path['nobs'].append(next_state)
            path['avg_backlog'].append(info['backlog'])
            #path['avg_backlog_change'].append(info['backlog_change'])
            #sasr.append((state, action, next_state, reward))
            total_reward += reward
            state = next_state
            if done:
                break
        paths.append(path)
        num_samples += len(paths[-1]['obs'])
    return paths, total_reward / num_samples#(num_trajectory * truncated_horizon)

def get_MSE(true_val, pred_vals):
    sq_error = np.square(np.array(true_val) - np.array(pred_vals))
    res = get_CI(sq_error)
    return res

# statistics/visualization related
def get_CI(data, confidence = 0.95):

    if (np.array(data) == None).all():
        return {}
    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.99:
        z = 2.576
    stats = {}
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    err = z * (std / np.sqrt(n))
    lower = mean - z * (std / np.sqrt(n))
    upper = mean + z * (std / np.sqrt(n))
    stats = {
        'mean': mean,
        'std': std,
        'lower': lower,
        'upper': upper,
        'err': err,
        'max': np.max(data),
        'min': np.min(data)
    }
    return stats

def plot_heatmap(pi, env, time):
    bounds = 51

    env_name = env.network_name
    if env_name == 'nmodel':
        types = ['both']
        cons_type = [[1, 1]]
    elif env_name == 'server_alloc':
        types = ['both', 'only1', 'only2']
        cons_type = [[1, 1], [1, 0], [0,1]]
        types = types[:1]
        cons_type = cons_type[:1]
    for idx, cons in enumerate(cons_type):
        ma = np.zeros((bounds, bounds))

        for q1 in range(0, bounds):
            for q2 in range(0, bounds):
                lens = np.array([q1, q2])
                if env_name == 'nmodel':
                    st = np.array(lens)
                elif env_name == 'server_alloc':
                    st = np.concatenate((lens, np.array(cons)))
                st = st.astype(np.float32)
                tr_st = torch.Tensor(env.transform_state(st))
                _, _, _, _, probs = pi.get_action_and_value(tr_st)

                probs = probs.probs.detach().numpy()

                q1_prob = probs[0]

                max_index = np.argmax(probs)
                q1_prob = 1. if max_index == 0 else 0
                # q1_prob = 1
                # if probs[0] 
                # prob = )[0] # serving queue 1
                ma[q1, q2] = q1_prob 

        ax = sns.heatmap(ma, linewidth=0.5)
        ax.invert_yaxis()
        plt.title('P(serving Q1)')
        plt.ylabel('Q1 length')
        plt.xlabel('Q2 length')
        #plt.imshow(ma, cmap = 'hot')
        #plt.savefig('{}_{}_{}_heat.pdf'.format(name, cons[0], cons[1]))
        plt.savefig('{}_{}_{}_heat.jpg'.format(time, env_name, types[idx]))

        plt.close()

# def plot_heatmap(pi, name, transformation = None, within_callback = False, env = None):
#     bounds = 51

#     env_name = env.get_attr('network_name')[0]
#     if env_name == 'nmodel':
#         types = ['both']
#         cons_type = [[1, 1]]
#     elif env_name == 'server_alloc':
#         types = ['both', 'only1', 'only2']
#         cons_type = [[1, 1], [1, 0], [0,1]]
#     for idx, cons in enumerate(cons_type):
#         ma = np.zeros((bounds, bounds))

#         for q1 in range(0, bounds):
#             for q2 in range(0, bounds):
#                 lens = np.array([q1, q2])
#                 if env_name == 'nmodel':
#                     st = np.array(lens)
#                 elif env_name == 'server_alloc':
#                     st = np.concatenate((lens, np.array(cons)))
#                 st = st.tolist()

#                 if within_callback:
#                     st = env.env_method('transform_state', st)[0]
#                     st = st.tolist()

#                 obs = torch.tensor([st])

#                 if hasattr(pi, 'policy') and isinstance(pi.policy, DQNPolicy):
#                     if pi.policy.boltzmann_exp:
#                         dis = pi.policy.get_distribution(obs)
#                         probs = dis.probs
#                         probs_np = probs.detach().numpy()[0]
#                         prob = probs_np[0]
#                     else:
#                         act = pi.policy.q_net.predict(obs)[0]
#                         # if action is queue 1, then prob 1
#                         prob = (act[0] == 0).astype(int)
#                 else:
#                     if within_callback:
#                         dis = pi.policy.get_distribution(obs)
#                     else:
#                         dis = pi.pi.policy.get_distribution(obs)
#                     probs = dis.distribution.probs
#                     probs_np = probs.detach().numpy()[0]
#                     prob = probs_np[0] # prob of serving queue 1
#                 ma[q1, q2] = prob 

#         ax = sns.heatmap(ma, linewidth=0.5)
#         ax.invert_yaxis()
#         plt.title('P(serving Q1)')
#         plt.ylabel('Q1 length')
#         plt.xlabel('Q2 length')
#         #plt.imshow(ma, cmap = 'hot')
#         #plt.savefig('{}_{}_{}_heat.pdf'.format(name, cons[0], cons[1]))
#         plt.savefig('{}_{}_heat.jpg'.format(name, types[idx]))

#         plt.close()

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def symlog(x, base = 'e'):
    if base == 'e':
        return np.sign(x) * np.log(np.abs(x) + 1)
    elif base == '10':
        return np.sign(x) * np.log10(np.abs(x) + 1)

def symsqrt(x):
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1)

def sigmoid(x):
    x = np.array(x)
    return 1. / (1. + np.exp(-x))

def tanh(x):
    return np.tanh(x)
