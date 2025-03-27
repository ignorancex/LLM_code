import numpy as np
import pandas as pd

from typing import List, Tuple

from scipy.stats import beta as beta_distribution
from scipy.optimize import differential_evolution

from trusts.model_dynamics import TrustDistribution

# binary performance values during learning stage experiments
BINARY_PERFORMANCE_LEARNING = [1, 1, -1, -1, -1, 1, -1, -1, -1, 1]
BINARY_PERFORMANCE_INFERENCE = [1, 1, -1, -1, -1]

TRAJECTORY_SIZE = 20


class MLEOptimization(object):

    def __init__(self,
                 learning_experiment_data: pd.DataFrame,
                 is_binary_performance: bool = False,
                 seed: int = 1773) -> object:
        
        self.random_seed = seed
        self.is_binary_performance = is_binary_performance
        self.learning_experiment_data = learning_experiment_data

        self.binary_performance_learning = BINARY_PERFORMANCE_LEARNING
        self.binary_performance_inference = BINARY_PERFORMANCE_INFERENCE

        if self.is_binary_performance:
            self.reward_binary = self.binary_performance_learning[0]

        # min-max bounds for optimizing parameters
        if self.is_binary_performance:
            # International Journal of Social Robotics, 2021
            # Article Link : https://link.springer.com/article/10.1007/s12369-020-00703-3
            self.bounds = [(1e-1, 1e+2), # initial_alpha
                           (1e-1, 1e+2), # initial_beta
                           (5e-4, 4e+2), # w_success
                           (5e-4, 4e+2)] # w_failure
        else:
            self.bounds = [(1e-1, 1e+2), # initial_alpha
                           (1e-1, 1e+2), # initial_beta
                           (5e-4, 4e+2), # w_success
                           (5e-4, 4e+2), # w_failure
                           (5e-3, 5e-1), # gamma
                          (-5e-1, 5e-1)] # epsilon_reward
    
    def negative_log_likelihood(self,
                                params: Tuple) -> float:
        
        if self.is_binary_performance:
            initial_alpha, initial_beta, w_success, w_failure = params
            gamma, epsilon_reward = 0.0, 0.0 # binary case scenarios (comparison model)
        else:
            initial_alpha, initial_beta, w_success, w_failure, \
                gamma, epsilon_reward = params
        
        self.trust_obj = TrustDistribution(initial_alpha=initial_alpha,
                                           initial_beta=initial_beta,
                                           initial_w_success=w_success,
                                           initial_w_failure=w_failure,
                                           gamma=gamma,
                                           epsilon_reward=epsilon_reward)
        
        # minimize sum of negative log likelihood
        log_likelihood = 0.0
        exp_number_idx = 0

        for _, row in self.learning_experiment_data.iterrows():

            time_step = row["TimeStep"]
            trust_label = row["TrustLabel"]
            reward = row["Reward"]

            if np.isnan(trust_label) or np.isnan(reward):
                continue
            
            # only consider reward data from the last time step of each trajectory
            if time_step != TRAJECTORY_SIZE:
                continue
            else:
                pass

            # in our particular case scenarios, reward is a notation of performance
            if self.is_binary_performance:
                reward = self.binary_performance_learning[exp_number_idx]
                self.reward_binary = reward
            
            # in a proposed case scenarios, reward function is continuous, therefore no need to change
            else:
                pass

            self.trust_obj.update_parameters(performance=reward)

            # check for invalid values of alpha and beta
            if np.isnan(self.trust_obj.alpha) or \
                np.isnan(self.trust_obj.beta) or \
                    self.trust_obj.alpha <= 0 or \
                        self.trust_obj.beta <= 0:
                # handle the case where alpha or beta becomes invalid
                continue
            
            else:
                nll = -np.log(beta_distribution.pdf(trust_label,
                                                    self.trust_obj.alpha,
                                                    self.trust_obj.beta))
                
                if nll == np.inf:
                    continue
                log_likelihood += nll

            if time_step == TRAJECTORY_SIZE:
                exp_number_idx += 1
            else:
                pass
        
        return log_likelihood
    
    def fit(self,
            initial_params: List[float]) -> Tuple:
        
        result = differential_evolution(func=self.negative_log_likelihood, # function to minimize
                                        x0=np.array(initial_params), # initial guess
                                        bounds=self.bounds, # bounds for the parameters
                                        seed=self.random_seed, # random seed
                                        init="latinhypercube", # initialization method
                                        strategy="best2bin", # strategy for selecting parents
                                        maxiter=25, # maximum iterations
                                        popsize=20, # population size
                                        tol=0.5, # tolerance for early stopping
                                        mutation=(0.5, 0.9), # mutation factor
                                        recombination=0.8, # cross-over probability
                                        workers=12, # number of workers (parallel)
                                        polish=True, # polish (L-BFGS-B) after optimization
                                        disp=False) # display the result after each iteration
        
        if self.is_binary_performance:
            mle_initial_alpha, mle_initial_beta, \
                mle_w_success, mle_w_failure = result.x
            
            return self.trust_obj, \
                mle_initial_alpha, mle_initial_beta, mle_w_success, mle_w_failure
        else:
            # extract the MLE parameters
            mle_initial_alpha, mle_initial_beta, mle_w_success, mle_w_failure, \
                mle_gamma, mle_epsilon_reward = result.x
            
            return self.trust_obj, \
                mle_initial_alpha, mle_initial_beta, mle_w_success, mle_w_failure, \
                    mle_gamma, mle_epsilon_reward
