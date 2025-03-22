from .testing_util import run_test
import torch
import re
import numpy as np
from transformers import Qwen2ForSequenceClassification, AutoModelForCausalLM
from transformers import AutoTokenizer

from .prm_utils import get_process_rewards, PromptType, PrefixesType


def phi(R_i, r_i, t, alpha_t_func, gamma, m=None):
    r"""
    Calculate the aggregated reward function \(\phi(R_i, r_i^{1:m})\).
    
    Parameters:
    - R_i: The final reward (scalar).
    - r_i: A sequence of intermediate rewards (array of length m).
    - t: The current timestep (scalar).
    - alpha_t_func: A function for the time-varying factor \(\alpha(t)\), which takes the time step t as input.
    - gamma: The discount factor (scalar, in the range [0, 1]).
    - m: The number of intermediate rewards (scalar).
    
    Returns:
    - Aggregated reward (scalar).
    """
    # Calculate the weighted sum of intermediate rewards, considering the discount factor
    if m is None:
        m = len(r_i)
    else:
        assert len(r_i) == m, "Number of intermediate rewards must match the given value of m."
    weighted_intermediate_rewards = np.sum([gamma**j * r_i[j] for j in range(m)])
    
    # Get the time-varying factor alpha(t)
    alpha_t = alpha_t_func(t)
    
    # Compute the aggregated reward using the given formula
    aggregated_reward = alpha_t * R_i + (1 - alpha_t) * (weighted_intermediate_rewards / m)
    
    return aggregated_reward

# Example: Define a linear decay function for alpha(t)
def linear_alpha(t, alpha_max=1.0, alpha_min=0.1, decay_rate=0.01):
    r"""
    A simple linear decay function for the time-varying factor \(\alpha(t)\).
    
    Parameters:
    - t: The current timestep (scalar).
    - alpha_max: The maximum value for \(\alpha(t)\) at t = 0 (default 1.0).
    - alpha_min: The minimum value for \(\alpha(t)\) as t increases (default 0.1).
    - decay_rate: The rate of decay (default 0.01).
    
    Returns:
    - The time-varying factor \(\alpha(t)\) (scalar).
    """
    return max(alpha_min, alpha_max - decay_rate * t)



class RewardAggregater():
    def __init__(self,model,tokenizer, phi_func=phi, alpha_func=linear_alpha, gamma=0.9, device='cuda'):
        
        self.device = device
        self.phi_func = phi_func
        self.alpha_func = alpha_func
        self.gamma = gamma

        self.model = model
        self.tokenizer = tokenizer
        
        self.device = torch.device(self.device)
        self.model.eval()  # Set model to evaluation mode
        
        self.reward = 0  # Initialize reward
        
        
    def compute_intermediate_rewards(self, prompts: PromptType, intermediate_texts: PrefixesType):
        process_rewards = get_process_rewards(
            self.model,
            self.tokenizer,
            prompts=prompts,
            completed_processes=intermediate_texts,
            tokenized_format='chat_completion',
        )
        # retrun (probability of good, probability of bad)
        process_rewards = [good_and_bad_probs[0] for good_and_bad_probs in process_rewards]
        return process_rewards
        
    
    def update_reward(self, prompt, intermediate_texts, in_outs, current_timestep, outcome_reward=None):

        # Compute intermediate rewards
        intermediate_rewards = self.compute_intermediate_rewards(prompt, intermediate_texts)
        
        # Number of intermediate rewards
        m = len(intermediate_rewards)
        if outcome_reward is None:
            outcome_reward = self.comupte_outcome_reward(intermediate_texts[-1], in_outs)
        # Calculate the aggregated reward
        self.reward = self.phi_func(
            R_i=outcome_reward,
            r_i=intermediate_rewards,
            t=current_timestep,
            alpha_t_func=self.alpha_func,
            gamma=self.gamma,
            m=m
        )
        
        return self.reward

    def exctract_runable_code(self, text):
        pattern = r"```python(.*?)```"
        
        try:
            code_snippets = re.findall(pattern, text, re.DOTALL)[-1]
        except:
            code_snippets = None
        
        return code_snippets
    
        
    def comupte_outcome_reward(self, final_step, in_outs):
        code = self.exctract_runable_code(final_step)
        if code == None:
            return 0
        result = run_test(code, in_outs, debug=True)
        
        score = [1 for r in result if r == True]
        score = sum(score)/len(result)
        return score
        
    def reset_reward(self):
        """
        Reset the aggregated reward to zero.
        """
        self.reward = 0

if __name__ == "__main__":
    ...