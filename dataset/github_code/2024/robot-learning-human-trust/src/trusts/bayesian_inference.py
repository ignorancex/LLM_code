import numpy as np

from scipy.stats import beta as beta_distribution
from scipy.optimize import minimize


# NOTE:
# Implementation for Baseline Comparisons
# "Modeling and Predicting Trust Dynamics in Human–Robot Teaming: A Bayesian Inference Approach", 
# International Journal of Social Robotics, 2021
# Article Link : https://link.springer.com/article/10.1007/s12369-020-00703-3


# Bayesian Inference based Trust Estimator Class
class BaselineTrustEstimator(object):
    
    def __init__(self,
                 alpha_0: float,
                 beta_0: float,
                 w_s: float,
                 w_f: float,
                 r: float = 0.8) -> object:
        
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        
        self.w_s = w_s
        self.w_f = w_f
        
        self.r = r
        
        self.alpha = alpha_0
        self.beta = beta_0
    
    def update_trust(self,
                     performance: int) -> None:
        
        # success
        if performance == 1:
            self.alpha += self.w_s
        
        # failure
        else:
            self.beta += self.w_f
    
    def estimate_trust(self):

        # suppress the specified warning
        with np.errstate(divide="ignore",
                         invalid="ignore"):
            
            beta_mean = self.alpha / (self.alpha + self.beta)
        
        return beta_mean

    def reset_params(self) -> None:
        
        self.alpha = self.alpha_0
        self.beta = self.beta_0


# Parameter Optimization via Maximum Likelihood Estimation
class PriorParameterOptimizer(object):
    
    def __init__(self,
                 agents_data: list) -> object:
        
        self.agents_data = agents_data

        self.bounds = [(1e-1, 5e+1), # initial_alpha
                       (1e-1, 5e+1), # initial_beta
                       (5e-4, 1e+0), # w_success
                       (5e-4, 6e+0)] # w_failure
    
    def estimate_priors(self) -> tuple:

        alpha_0_values, beta_0_values, w_s_values, w_f_values = [], [], [], []
        
        for agent_data in self.agents_data:
            agent_params = self._estimate_agent_params(trust_history=agent_data["trust_history"],
                                                       performance_history=agent_data["performance_history"])
            
            alpha_0_values.append(agent_params["alpha_0"])
            beta_0_values.append(agent_params["beta_0"])
            w_s_values.append(agent_params["w_s"])
            w_f_values.append(agent_params["w_f"])
        
        alpha_0_prior = self._estimate_prior(values=alpha_0_values)
        beta_0_prior = self._estimate_prior(values=beta_0_values)
        w_s_prior = self._estimate_prior(values=w_s_values)
        w_f_prior = self._estimate_prior(values=w_f_values)
        
        return alpha_0_prior, beta_0_prior, w_s_prior, w_f_prior

    def _estimate_agent_params(self,
                               trust_history: list,
                               performance_history: list) -> dict:
        
        alpha_0, beta_0, w_s, w_f = 0.5, 0.5, 0.2, 0.2
        
        def objective_fun(params: list) -> float:
            
            alpha, beta, w_s, w_f = params
            
            likelihoods = beta_distribution.pdf(trust_history,
                                                alpha,
                                                beta)
            
            log_likelihoods = np.log(likelihoods)

            # check for NaNs in log likelihoods
            if np.any(np.isnan(log_likelihoods)):
                return np.inf
            
            nll = -np.sum(log_likelihoods * (1 - performance_history) + \
                          np.log(1 - likelihoods) * performance_history)
            
            # check for infinity in the negative log likelihood
            if np.isinf(nll):
                return np.inf
            
            return nll
        
        result = minimize(objective_fun,
                          [alpha_0,
                           beta_0,
                           w_s,
                           w_f],
                          bounds=self.bounds)
        
        alpha_0, beta_0, w_s, w_f = result.x
        
        param_set = {"alpha_0": alpha_0,
                     "beta_0": beta_0,
                     "w_s": w_s,
                     "w_f": w_f}
        
        return param_set
    
    def _estimate_prior(self,
                        values: list) -> tuple:
        
        mu = np.mean(values)
        std = np.std(values)
        
        return mu, std


# Main Testing Script with Dummy Data
if __name__ == "__main__":

    num_agents = 10
    num_interactions = 100
    
    # create a trust estimator for a new agent
    robot_reliability = 0.8

    agents_data = []

    # create a dummy data from uniform distribution
    for agent_id in range(num_agents):
        
        trust_history = np.random.uniform(0,
                                          1,
                                          num_interactions)
        
        # assume robot reliability = 0.8 [in the paper, it is set as %70, %80, and %90]
        performance_history = np.random.binomial(1,
                                                 robot_reliability,
                                                 num_interactions)
        
        agents_data.append({"trust_history": trust_history,
                            "performance_history": performance_history})
    
    # estimate prior distributions for parameters
    prior_optimizer = PriorParameterOptimizer(agents_data=agents_data)

    alpha_0_prior, beta_0_prior, w_s_prior, w_f_prior = prior_optimizer.estimate_priors()

    trust_predictor = BaselineTrustEstimator(alpha_0=alpha_0_prior[0],
                                             beta_0=beta_0_prior[0],
                                             w_s=w_s_prior[0],
                                             w_f=w_f_prior[0],
                                             r=robot_reliability)
    
    # simulate interactions with the new agent
    trust_history = []
    performance_history = []

    for i in range(num_interactions):
        
        # observe the robot's performance
        performance = np.random.binomial(1,
                                         robot_reliability)
        performance_history.append(performance)
        trust_predictor.update_trust(performance)
        predicted_trust = trust_predictor.estimate_trust()
        trust_history.append(predicted_trust)
        
        print("\n")
        print("Predicted Trust History:", trust_history)
        print("Performance History:", performance_history)
