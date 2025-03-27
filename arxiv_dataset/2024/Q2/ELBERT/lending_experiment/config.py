'''
Only things that won't be changed are hard-wired here.
Things like N_location, which we might change, are specified in the main file
'''
import torch

########## Experiment Setup Parameters ##########
EXP_DIR = './experiments/'

########## Env Parameters ##########
DELAYED_IMPACT_CLUSTER_PROBS = (
    (0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.0),
    (0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.0),
)
# NUM_GROUPS = 2
GROUP_0_PROB = 0.5
BANK_STARTING_CASH= 10000
INTEREST_RATE = 1
CLUSTER_SHIFT_INCREMENT= 0.01
CLUSTER_PROBABILITIES = DELAYED_IMPACT_CLUSTER_PROBS

# ########## Training Parameters ##########
POLICY_KWARGS_fair = dict(activation_fn=torch.nn.ReLU,
                     net_arch = dict(vf=[256, 128], pi=[256, 128])) # new sb3: shared layers is not defined in net_arch here. 
SAVE_FREQ = 1e8  # save frequency in timesteps: don't save model for now
EP_TIMESTEPS = 2048 # Number of steps per episode for training

########## Evaluation ##########
EVAL_NUM_EPS = 3 # number of episodes for evaluation: APPO's paper uses 10
EVAL_INTERVAL = 5 # number of training rollout per evaluation (1 training rollout is BUFFER_SIZE_TRAINING samples)
EP_TIMESTEPS_EVAL = 10000 # Number of steps per episode for evaluation

