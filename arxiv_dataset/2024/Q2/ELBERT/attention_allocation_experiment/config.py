'''
Only things that won't be changed are hard-wired here.
Things like N_location, which we might change, are specified in the main file
'''
import torch

########## Experiment Setup Parameters ##########
EXP_DIR = './experiments/'

########## Training Parameters ##########
POLICY_KWARGS_fair = dict(activation_fn=torch.nn.ReLU,
                     net_arch = dict(vf=[128, 64], pi=[128, 64])) # new sb3: shared layers is not defined in net_arch here. 
SAVE_FREQ = 1e8  # save frequency in timesteps: don't save model for now
EP_TIMESTEPS = 1024 # Number of steps per episode for training

########## Evaluation ##########
EVAL_NUM_EPS = 10 # number of episodes for evaluation
EVAL_INTERVAL = 5 # number of training rollout per evaluation (1 training rollout is BUFFER_SIZE_TRAINING samples)
EP_TIMESTEPS_EVAL = 1024 # Number of steps per episode for evaluation

