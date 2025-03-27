import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

import pdb
from matplotlib import rcParams
import os
import itertools
import yaml

from scipy.stats import binom

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

import plot_custom_utils

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def bool_argument(parser, name, default=False, msg=''):
    dest = name.replace('-', '_')
    parser.add_argument('--%s' % name, dest=dest, type=bool, default=default, help=msg)
    parser.add_argument('--no-%s' % name, dest=dest, type=bool, default=default, help=msg)

parser = argparse.ArgumentParser()
parser.add_argument('result_directory', help=help)
parser.add_argument('--env_name', type = str)
parser.add_argument('--tr_metric', type = str, default = 'avg_backlog')
parser.add_argument('--type', type = str, choices = ['iterations', 'final'], default = 'iterations')
parser.add_argument('--stat', type = str)
parser.add_argument('--y_label', type = str, default = None)
parser.add_argument('--y_log', type = str2bool, default = False)

FLAGS = parser.parse_args()

def plot_vs_iterations(data, file_name, plot_params, metric = 'err', interval_type = 'tolerance'):
    algorithms = sorted(list(data.keys())) 
    colors = sns.color_palette('colorblind')
    xlabels = algorithms
    #color_idxs = [0, 3, 4, 2, 1, 7, 8, 5, 6, 9][:len(algorithms)]
    color_idxs = [0, 1, 2, 3, 4, 5, 7, 8, 9][:len(algorithms)]
    color_dict = dict(zip(xlabels, [colors[idx] for idx in color_idxs]))

    num_x_ticks = 0
    metric_val_dict = {}
    for algo in algorithms:
        m_vals = np.array(data[algo][metric])
        indx = np.argsort(m_vals[:, -1])
        sd = m_vals[indx]
        nu = len(indx)
        m_vals = sd[nu // 4: nu*3//4, :]
        m_vals = np.expand_dims(m_vals, axis = 1) 
        metric_val_dict[algo] = m_vals
        num_x_ticks = m_vals.shape[-1]

    times = np.array([i for i in range(num_x_ticks)]) 

    times_metrics_dict = {algorithm: metric_val[:, :, times] for algorithm, metric_val
                              in metric_val_dict.items()}
    mean = lambda m_vals: np.array([metrics.aggregate_mean(m_vals[..., time])
                                   for time in range(m_vals.shape[-1])])

    if interval_type == 'tolerance':
        mean_m_vals, mean_cis = plot_custom_utils.get_tolerance_interval(times_metrics_dict)
    elif interval_type == 'strat_boot':
        mean_m_vals, mean_cis = rly.get_interval_estimates(
            times_metrics_dict, mean, reps=500)#0)
    elif interval_type == 'student':
        mean_m_vals, mean_cis = plot_custom_utils.get_student_interval(times_metrics_dict)
    ax = plot_utils.plot_sample_efficiency_curve(
        times+1, mean_m_vals, mean_cis, algorithms=algorithms,
        figsize = (15, 10),
        xlabel=r'Interaction Steps (x10)',
        ylabel=metric if FLAGS.y_label is None else FLAGS.y_label,
        ticklabelsize = 40,
        labelsize = 40,
        colors = color_dict)

    fake_patches = [mpatches.Patch(color=color_dict[alg], 
                                   alpha=0.75) for alg in algorithms]
    legend = plt.legend(fake_patches, algorithms, loc='best', 
                        fancybox=True, ncol=2,#len(algorithms), 
                        fontsize='40')#xx-large')
    #ax.set_yscale('log')
    ax.set_ylim(plot_params['y_range'])
    #plt.legend(fontsize = 20, loc = 'best')
    plt.tight_layout()
    plt.savefig('{}.jpg'.format(file_name))
    plt.close()

def collect_data():

    data = {}
    for basename in os.listdir(FLAGS.result_directory):
        if '.npy' not in basename:
            continue
        if '.npz' in basename:
            continue
        f_name = os.path.join(FLAGS.result_directory, basename)
        # 'results_queue_main_mdp_2/env_queue_exp_main_algo_STOP-1.5_seed_553395_mdp-num_2_truncated-horizon_200_lr_0.0003_epochs_10_adam-beta_0.9.npy'
        names = f_name.split('_')
        summary = np.load(f_name, allow_pickle = True).item()
        truncated_horizon = summary['hp']['truncated_horizon']
        replay_epochs = summary['hp']['replay_epochs']
        batch_size = summary['hp']['batch_size']
        lr = float(summary['hp']['lr'])
        adam_beta = -1#float(summary['hp']['adam_beta'])
        hp = (truncated_horizon, lr, replay_epochs, batch_size, adam_beta)
        results = summary['results']
        algos = summary['results'].keys()
        
        algo_name = list(algos)[0]

        #if algo_name not in set(['MW', 'PPO', 'STOP-C']):
        #    continue

        for algo in algos:

            label = algo
            if algo == 'STOP-C':
                label = 'STOP-3'
            elif algo == 'STOP-Q':
                label = 'STOP-2'
            elif algo == 'STOP-L':
                label = 'STOP-1'

            if label not in data:
                data[label] = {
                    'avg_backlog': [],
                    'learning_backlog': [],
                    'learning_timesteps': [],
                    'unstable_frac': [],
                    'unstable_adv_mean': []
                }
            data[label]['avg_backlog'].append(results[algo]['avg_backlog'] if 'avg_backlog' in results[algo] else 0)
            data[label]['unstable_frac'].append(results[algo]['unstable_frac'] if 'unstable_frac' in results[algo] else 0)
            data[label]['unstable_adv_mean'].append(results[algo]['unstable_adv_mean'] if 'unstable_adv_mean' in results[algo] else 0)
    return data

def main():
    nice_fonts = {
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 20,
        "font.size": 20,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 16,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }
    #plt.style.use('seaborn')

    plot_params = {'bfont': 45,
               'lfont': 45,
               'tfont': 45,
               'legend': True,
               'legend_loc': 0,
               'legend_cols': 2,
               #'y_range': (90, 1000),
               'y_range': None,#(10, 20),
               'x_range': None,
               'log_scale': False,
               #'y_label': r'(relative) MSE($\rho(\pi_e)$)',
               'y_label': FLAGS.y_label,
               #'y_label': '(relative) MSE',
               'shade_error': True,
               'x_mult': 1,
               'axis_label_pad': 15}

    fname = '{}_'.format(FLAGS.env_name)
    file_name = fname + '{}_{}'.format(FLAGS.tr_metric, FLAGS.stat)

    data = collect_data()
    plot_vs_iterations(data, file_name, plot_params, metric = FLAGS.tr_metric, interval_type = FLAGS.stat)

        
if __name__ == '__main__':
    main()

