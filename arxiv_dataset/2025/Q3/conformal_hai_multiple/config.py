import numpy as np
import torch
import os
import argparse
import datetime

class Config:
    def __init__(self) -> None:
        pass
    
parser = argparse.ArgumentParser()
parser.add_argument("--n_labels", type=int, default=10, help='Size of label space in synthetic prediction tasks') 
parser.add_argument("--cal_split", type=float, help='Fraction of data to be used for the estimation and calibration set') 
parser.add_argument("--runs", type=int, default=10, help='Number of repetitions of each experiment')
parser.add_argument("--topk", type=int, default=5, help='Set "k" for prediction sets with top-k labels')
parser.add_argument('--y_humans_cnt', type=int, required=True, help='Number of experts')
parser.add_argument('--human_subset_select', type=str, required=True, help='Type of subset selection (e.g., "greedy", "random", "all")')
parser.add_argument('--alpha_index', type=int, required=True, help='Description of alpha_index')
parser.add_argument('--CM_lapl_smoothing', action='store_true', help='Whether to apply CM Laplace smoothing (default: False)')
parser.add_argument('--CM_lapl_param', type=float, default=0, help='CM Laplace smoothing parameter')

args,unknown = parser.parse_known_args()
conf = Config()

conf.ROOT_DIR = os.path.dirname(__file__)

# if torch.cuda.is_available():
#     conf.device = torch.cuda.current_device()
# else:
conf.device = 'cpu'


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

conf.base_folder =   "/home/hapaat/"
conf.y_humans_init = 'multiple'                                     # 'single', 'multiple', None                    # Simulate humans  # Change often
conf.y_humans_cnt = args.y_humans_cnt                               # ignored if conf.y_humans_init == 'single'
conf.human_subset_select = args.human_subset_select                 # can be all, random, greedy    # important change
conf.alpha_index = args.alpha_index

conf.sim_humans_select = 'mode'                                 # can be 'mode' or 'weighted'; IGNORED if conf.y_humans_init == 'single'
conf.expert_matrix = ["h_based_ws_t"]                           # "h_based_ws_t", "full_cm_ws_t", "orig"  # 'orig' has access to the GT as in Straitouri et al.
conf.calc_metric_inference = True

conf.enable_find_m = False      # change
conf.enable_find_alpha = False
conf.accuracy = None                                    # changed                       # set value for 'synthetic' setting
conf.uncertainty_h = False                              # Uncertainty-aware expert      # will take effect if y_humans_init = 'single'
conf.conformalized_expert = None                        # "simulation", "all", None        # change

conf.CM_lapl_smoothing = args.CM_lapl_smoothing         # Set to True to create more inaccurate human simulations
conf.CM_lapl_param = args.CM_lapl_param
conf.lapl_smoothing = False                             # Set to True to create more inaccurate human simulations
conf.lapl_param = 0.020                                 # Degree of human simulation inaccuracy; ignored if conf.lapl_smoothing is False
conf.model_lapl = False
conf.model_lapl_param = 0.10

# Expert
conf.ignore_test = True                                 # ignore test data for the estimation of the confusion matrix
conf.expert_type = ["orig"]                             # orig, median

# For plotting
conf.plot_alpha_vs_sets = False
conf.plot_alpha_vs_succprob = False
time = datetime.datetime.now().time().strftime("%H:%M:%S")
conf.path_plot_alpha_vs_sets = f"{conf.base_folder}epcp/alpha_vs_sets_{datetime.datetime.now().date()}_{time}.txt" 
conf.path_plot_alpha_vs_succprob = f"{conf.base_folder}epcp/alpha_vs_succprob_{datetime.datetime.now().date()}_{time}.txt" 

# # IGNORE THESE SETTINGS
# conf.calest_portion = 0.70                            
# conf.test = 0.70
# conf.val = 0.10
# conf.pretrained_model_path = conf.base_folder + "epcp/results_imagenet/pretrained_models/vgg19_epoch10_fully_trained.pth.tar"

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------




conf.seed = 12345678
conf.torch_rng = torch.Generator(device=conf.device).manual_seed(conf.seed)
conf.rng = np.random.default_rng(seed=conf.seed)

# Dataset size
conf.data_size = 10000 
# Parameter to control difficulty of the dataset in synthetic experiments # NOTE Hyperparameter? 
conf.class_sep = {10:{0.3:0.46, 0.5:1.09, 0.7:1.72, 0.9: 2.75}, 
                  50:{0.3:1.31, 0.5:2.16, 0.7:3.19, 0.9: 5.27},
                 100:{0.3:1.75, 0.5:2.8, 0.7:4.4, 0.9: 7.7}}

conf.accuracies = np.arange(3,10, 2)/10.
conf.is_oblivious = False # If set, human predicts labels at random

conf.n_labels = args.n_labels
conf.cal_split = args.cal_split

conf.test_split = 0.10                       # Test split for synthetic data experiments
conf.n_runs_per_split = args.runs
conf.delta = 0.1

# Synthetic data label distribution
distr = conf.rng.dirichlet(np.ones(conf.n_labels),size=1)
sum_distr = distr.sum()
if sum_distr < 1.:
    distr += (1 - sum_distr)/conf.n_labels
conf.class_probabilities = distr

# Names of classifiers used in real data experiments # For CIFAR10H only
conf.model_names = ['densenet-bc-L190-k40', 'preresnet-110', 'resnet-110']    #'cnn_data', 'r_low_acc'   # 'densenet-bc-L190-k40', 'preresnet-110', 'resnet-110', "cnn_data", 'r_low_acc'