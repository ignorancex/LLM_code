from __future__ import print_function
from __future__ import division

import sys
import os
import argparse
import subprocess
import random
import time
import pdb
import itertools
import numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
# saving
parser.add_argument('result_directory', default = None, help='Directory to write results to.')

# common setup
parser.add_argument('--env_name', type = str, required = True)
parser.add_argument('--mdp_num', default = 0, type = int, required = True)
#parser.add_argument('--gamma', default = 0.999, type = float)
parser.add_argument('--deployed_interaction_steps', default = 250000, type = int)
parser.add_argument('--pretrained_name', default = None, type = str)
parser.add_argument('--anneal_lr', default = False, type = str2bool)
parser.add_argument('--state_bound', default = np.inf, type = float)

parser.add_argument('--training_horizon', default = 10000, type = int)
parser.add_argument('--train_interaction_steps', default = 200000, type = int)
parser.add_argument('--train_and_test', default = False, type = str2bool)

# variables
parser.add_argument('--num_trials', default = 1, type=int, help='The number of trials to launch.')
parser.add_argument('--condor', default = False, action='store_true', help='run experiments on condor')
parser.add_argument('--exp_name', default = 'gan', type = str)

FLAGS = parser.parse_args()
ct = 0
EXECUTABLE = 'exp.sh'

def get_cmd(seed,
            outfile,
            truncated_horizon,
            lr,
            replay_epochs,
            adam_beta,
            algo_info,
            condor = False):
  
    algo_name = algo_info[0]

    arguments = '--outfile %s --seed %d' % (outfile, seed)
  
    arguments += ' --exp_name %s' % FLAGS.exp_name
    arguments += ' --env_name %s' % FLAGS.env_name
    arguments += ' --algo_name %s' % algo_name
    
    if 'PPO' in algo_name or 'STOP' in algo_name :
        arguments += ' --reward_function %s' % algo_info[1]
        arguments += ' --state_transformation %s' % algo_info[2]
        arguments += ' --lyp_power %s' % (algo_info[3] if algo_info[3] is not None else str(0))
        if 'AM' in algo_name:
            arguments += ' --use_action_mask True'
    
    arguments += ' --mdp_num %d' % FLAGS.mdp_num
    arguments += ' --deployed_interaction_steps %d' % FLAGS.deployed_interaction_steps
    if FLAGS.pretrained_name:
        arguments += ' --pretrained_name %s' % FLAGS.pretrained_name
    
    arguments += ' --adam_beta %f' % adam_beta
    arguments += ' --anneal_lr %s' % FLAGS.anneal_lr

    arguments += ' --truncated_horizon %d' % truncated_horizon
    arguments += ' --replay_epochs %d' % replay_epochs
    arguments += ' --lr %f' % lr
    arguments += ' --state_bound %f' % FLAGS.state_bound

    arguments += ' --training_horizon %d' % FLAGS.training_horizon
    arguments += ' --train_interaction_steps %d' % FLAGS.train_interaction_steps
    arguments += ' --train_and_test %s' % FLAGS.train_and_test

    if FLAGS.condor:
        cmd = '%s' % (arguments)
    else:
        EXECUTABLE = 'run_single_continual.py'
        cmd = 'python3 %s %s' % (EXECUTABLE, arguments)
    return cmd

def run_trial(seed,
            outfile,
            truncated_horizon,
            lr,
            replay_epochs,
            adam_beta,
            algo_info,
            condor = False):

    cmd = get_cmd(seed,
                outfile,
                truncated_horizon,
                lr,
                replay_epochs,
                adam_beta,
                algo_info)
    if condor:
        if FLAGS.env_name == 'traffic':
            submitFile = 'universe = container\n'
            submitFile += 'container_image = http://proxy.chtc.wisc.edu/SQUID/pavse/sumo.sif\n'
        else:
            submitFile = 'universe = vanilla\n'
        submitFile += 'executable = ' + EXECUTABLE + "\n"
        submitFile += 'arguments = ' + cmd + '\n'
        submitFile += 'error = %s.err\n' % outfile
        #submitFile += 'log = %s.log\n' % outfile
        submitFile += 'log = /dev/null\n'
        submitFile += 'output = /dev/null\n'
        #submitFile += 'output = %s.out\n' % outfile
        submitFile += 'should_transfer_files = YES\n'
        submitFile += 'when_to_transfer_output = ON_EXIT\n'

        setup_files = 'http://proxy.chtc.wisc.edu/SQUID/pavse/myenv.tar.gz'
        common_main_files = 'run_single_continual.py, policies.py, utils.py, algos, pretrained_policies, cleanrl_algo'

        if FLAGS.env_name == 'traffic':
            domains = 'run_traffic.py, sumo'
            submitFile += 'transfer_input_files = http://proxy.chtc.wisc.edu/SQUID/pavse/sumo.sif, {}, {}, {}\n'.format(setup_files, common_main_files, domains)
        else:
            domains = 'server_allocation.py, gridworld.py, nmodel.py'
            submitFile += 'transfer_input_files = {}, {}, {}\n'.format(setup_files, common_main_files, domains)
        submitFile += 'requirements = (has_avx == True)\n'
        submitFile += 'request_cpus = 1\n'
        submitFile += 'request_memory = 7GB\n'
        submitFile += 'request_disk = 8GB\n'
        submitFile += 'queue'

        proc = subprocess.Popen('condor_submit', stdin=subprocess.PIPE)
        proc.stdin.write(submitFile.encode())
        proc.stdin.close()
        time.sleep(0.2)
    else:
        # TODO
        pdb.set_trace()
        #subprocess.run('"conda init bash; conda activate research; {}"'.format(cmd), shell=True)
        #cmd = 'bash -c "source activate root"' 
        subprocess.Popen(('conda run -n research ' + cmd).split())

def _launch_trial(seeds, t, lr, replay_epochs, adam_beta, algo_info):

    algo_name = algo_info[0]
    global ct
    for seed in seeds: 
        outfile = 'env_{}_exp_{}_algo_{}_seed_{}_mdp-num_{}_truncated-horizon_{}_lr_{}_epochs_{}_adam-beta_{}.npy'\
            .format(FLAGS.env_name,
            FLAGS.exp_name, algo_name, seed,
            FLAGS.mdp_num, 
            t, 
            lr, 
            replay_epochs,
            adam_beta)

        if os.path.exists(outfile):
            continue
        run_trial(seed,
                outfile,
                truncated_horizon = t,
                lr = lr,
                replay_epochs = replay_epochs,
                adam_beta = adam_beta,
                algo_info = algo_info,
                condor = FLAGS.condor)
        ct += 1
        print ('submitted job number: %d' % ct)

def main():  # noqa
    if FLAGS.result_directory is None:
        print('Need to provide result directory')
        sys.exit()
    directory = FLAGS.result_directory + '_' + FLAGS.env_name + '_' + FLAGS.exp_name + '_mdp_' + str(FLAGS.mdp_num)

    if not os.path.exists(directory):
        os.makedirs(directory)
    seeds = [random.randint(0, 1e6) for _ in range(FLAGS.num_trials)]

    # length of trajectory used to train RL policy
    truncated_horizon = [200]
    replay_epoch = [10]
    lrs = [3e-4]
    adam_betas = [0.9]
    
    # (algo name, reward_func, state_transformation, normalize)
    rl_algos = [
                    #('PPO', 'opt', 'id', None),
                    #('STOP-L', 'stab', 'symloge', 1.),
                    #('STOP-1.5', 'stab', 'symloge', 1.5),
                    #('STOP-Q', 'stab', 'symloge', 2.),
                    #('STOP-2.5', 'stab', 'symloge', 2.5),
                    #('STOP-C', 'stab', 'symloge', 3.),
                    #('STOP-4', 'stab', 'symloge', 4.),
                    #('STOP-5', 'stab', 'symloge', 5.),

                    #('STOP-ID', 'stab', 'id', 3.),
                    #('STOP-SIG', 'stab', 'sigmoid', 3.),
                    #('STOP-SL', 'stab', 'symloge', 3.),
                    #('STOP-SS', 'stab', 'symsqrt', 3.),

                    ('STOP-ID', 'opt', 'id', None),
                    ('STOP-SIG', 'opt', 'sigmoid', None),
                    ('STOP-SL', 'opt', 'symloge', None),
                    ('STOP-SS', 'opt', 'symsqrt', None),
                ]

    rl_combined = [truncated_horizon, lrs, replay_epoch, adam_betas, rl_algos]
    rl_combined = list(itertools.product(*rl_combined))
    
    heur_algos = [
        #('Thresh',),
        ('MW',) if FLAGS.env_name == 'queue' else ('MWN',),
    ]

    heur_combined = [[0], [0], [0], [0], heur_algos]
    heur_combined = list(itertools.product(*heur_combined))

    #all_combined = heur_combined
    all_combined = rl_combined
    #all_combined = rl_combined + heur_combined

    for e in all_combined:
        th, lr, rep_epoch, adam_beta, algo_info = e
        _launch_trial(seeds, th, lr, rep_epoch, adam_beta, algo_info) # setting batch_size to th

    print('%d experiments ran.' % ct)

if __name__ == "__main__":
    main()


