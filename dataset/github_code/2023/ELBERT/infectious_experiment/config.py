'''
Only things that won't be changed are hard-wired here.
Things like N_location, which we might change, are specified in the main file

NOTE: BUFFER_SIZE_TRAINING needs to be a multiply of EP_TIMESTEPS_EVAL right now (a buffer is a multiply of episode)
Otherwise there will be errors
'''
import torch

########## Experiment Setup Parameters ##########
EXP_DIR = './experiments/'

########## Training Parameters ##########
POLICY_KWARGS_fair = dict(activation_fn=torch.nn.ReLU,
                     net_arch = dict(vf=[512, 256], pi=[512, 256])) # new sb3: shared layers is not defined in net_arch here. 
SAVE_FREQ = 1e8  # save frequency in timesteps: don't save model for now
EP_TIMESTEPS = 20 # Number of steps per episode for training

########## Evaluation ##########
EVAL_NUM_EPS = 200 # number of episodes for evaluation
EVAL_INTERVAL = 5 # number of training rollout per evaluation (1 training rollout is BUFFER_SIZE_TRAINING samples)
EP_TIMESTEPS_EVAL = 20 # Number of steps per episode for evaluation

########## Env Parameters ##########
GRAPH_NAME = 'karate'
BURNIN = 1

