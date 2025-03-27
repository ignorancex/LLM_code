import numpy as np

from typing import List, Optional
from scipy import stats


class TrustDistribution(object):

    def __init__(self,
                 initial_alpha : float = 0.5,
                 initial_beta : float = 0.5,
                 initial_w_success: float = 0.2,
                 initial_w_failure: float = 0.2,
                 gamma: float = 0.1,
                 epsilon_reward: float = 0.1) -> object:
        
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta

        self.alpha = initial_alpha
        self.beta = initial_beta

        self.w_success = initial_w_success
        self.w_failure = initial_w_failure
        
        self.gamma = gamma
        self.epsilon_reward = epsilon_reward
        
        self.list_alpha = [self.alpha]
        self.list_beta = [self.beta]

        self.list_w_success = [self.w_success]
        self.list_w_failure = [self.w_failure]

        # making sure that there are no suspicious values for trust
        self.success_trust_threshold = 0.1
        self.failure_trust_threshold = 0.9

        # taken as a half of tthe scenario length after observations
        self.history_buffer_length = 100

        self.update_beta_distribution()
    
    def update_parameters(self,
                          performance: float) -> None:
        
        # update only alpha parameter
        if performance > self.epsilon_reward:

            # alpha parameter upgrade and extend the list of alpha parameters
            alpha_history = self.cumulative_value(self.list_alpha[-self.history_buffer_length :])
            self.alpha =  alpha_history + self.w_success * performance
            self.list_alpha.append(self.alpha)

            # beta parameter will be the same as before and
            # no need to extend the list of beta parameters
            self.beta = self.cumulative_value(self.list_beta[-self.history_buffer_length :])

        # update only beta parameter
        else:

            # beta parameter upgrade and extend the list of beta parameters
            beta_history = self.cumulative_value(self.list_beta[-self.history_buffer_length :])
            self.beta = beta_history + self.w_failure * float(np.exp(abs(performance)))
            self.list_beta.append(self.beta)

            # alpha parameter will be the same as before and
            # no need to extend the list of alpha parameters
            self.alpha = self.cumulative_value(self.list_alpha[-self.history_buffer_length :])
        
        # update the underlying beta distribution based on current alpha and beta
        self.update_beta_distribution()
    
    def update_weights(self,
                       w_success: Optional[float] = None,
                       w_failure: Optional[float] = None) -> None:
        
        if w_success is not None:
            self.w_success = abs(w_success)
            self.list_w_success.append(w_success)
        
        if w_failure is not None:
            self.w_failure = abs(w_failure)
            self.list_w_failure.append(w_failure)
    
    def cumulative_value(self,
                         parameter_list: List[float]) -> None:
        
        num = len(parameter_list)

        # consider the last value only as a history when gamma is zero
        if self.gamma == 0.0:
            return parameter_list[-1]

        history = 0.0
        for idx in range(num - 1, - 1, - 1):
            history += (self.gamma ** (num - 1 - idx)) * parameter_list[idx]
        
        return history
    
    def change_success_weight(self,
                              true_value: float,
                              last_performance: float) -> None:
        
        if true_value < self.success_trust_threshold:
            raise ValueError("Error: trust measurement should be greater than or equal to success_trust_threshold for success.")
        
        alpha_history = self.cumulative_value(self.list_alpha)

        w_success = ((true_value * (alpha_history + self.beta)) - alpha_history) / (last_performance * (1.0 - true_value + 1e-2))
        self.update_weights(w_success=w_success)
    
    def change_failure_weight(self,
                              true_value: float,
                              last_performance: float) -> None:
        
        if true_value > self.failure_trust_threshold:
            raise ValueError("Error: trust measurement should be less than or equal to failure_trust_threshold for failure.")
        
        beta_history = self.cumulative_value(self.list_beta)

        w_failure = (self.alpha - (true_value * (self.alpha + beta_history))) / ((true_value + 1e-2) * np.exp(np.abs(last_performance)))
        self.update_weights(w_failure=w_failure)
    
    def get_trust_level(self) -> float:

        return self.distribution.mean()
    
    def get_beta_distribution_mean(self) -> float:

        return self.alpha / (self.alpha + self.beta)
    
    def update_beta_distribution(self) -> None:

        self.distribution = stats.beta(self.alpha,
                                       self.beta)
