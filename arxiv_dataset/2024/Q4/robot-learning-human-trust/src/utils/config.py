import os

from datetime import datetime


class Config(object):

    def __init__(self) -> object:

        today = datetime.today() # month - date - year
        now = datetime.now() # hours - minutes - seconds

        current_date = str(today.strftime("%b_%d_%Y"))
        current_time = str(now.strftime("%H_%M_%S"))

        # month_date_year-hour_minute_second
        self.time_info = current_date + "-" + current_time

        print("Current Time: ", self.time_info)
    
    def parameters(self):
        
        self.seed = 1773
        self.device = "cpu"

        # neural network optimization parameters
        self.state_size = 3
        self.hidden_size = 64
        self.action_size = 7
        self.reward_size = 1

        # strategy for batch sampling
        self.batch_size_policy = 32
        self.batch_size_reward = 128
        self.data_shuffle = True
        self.num_workers = 0

        # min-max log standard deviations for policy network output
        self.policy_log_std_min = -24 # exp(-24.0) = 0.0
        self.policy_log_std_max = 2.4 # exp(2.4) = 11.0
        
        # learning rates
        self.policy_lr = 1e-4
        self.reward_lr = 5e-4
        
        # optimizer scheduler parameters
        self.policy_scheduler_step_size = 1000
        self.reward_scheduler_step_size = 1000
        self.policy_scheduler_gamma = 0.9
        self.reward_scheduler_gamma = 0.9
        self.policy_weight_decay = 1e-4
        self.reward_weight_decay = 1e-4

        # 15% of the dataset will be used for validation
        self.validation_split = 0.15

        # patience for early stopping (number of epochs with no improvement on validation loss)
        self.early_stopping_patience = 20

        # maximum size of replay buffer
        self.replay_buffer_capacity = 10000

        # continuous beta distibution parameters
        self.initial_alpha = 0.50651927199982 # MLE optimized results
        self.initial_beta = 0.992862478025096 # MLE optimized results

        # weights for success and failure updates
        self.initial_w_success = 3.78971727548625 # MLE optimized results
        self.initial_w_failure = 4.53965297545633 # MLE optimized results

        # discount factor to store cumulative history of trust values
        self.gamma = 0.005 # MLE optimized results

        # threshold of reward value for diferentiating success and failure
        self.epsilon_reward = 0.378482948006227 # MLE optimized results
    
    def model_saving_path(self,
                          directory: str) -> str:
        
        if not isinstance(directory, str):
            raise TypeError("Input 'directory' in model_saving_path function must be a string.")
        if not os.path.exists(directory):
            raise ValueError("Input 'directory' in model_saving_path function must be a valid path.")
        
        saving_path = os.path.join(directory, self.time_info)

        # create directory if does not exist
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        
        return saving_path
