import logging
import os
from collections import OrderedDict
from types import SimpleNamespace
import time 
import torch

from tpsr.reward import compute_reward_nesymres

class RLEnv:
    """
    Equation Generation RL environment.

    State: a list of tokens.
    Action: a token (an integer).
    Reward: Fittness reward of the generated equation.
    """
    def __init__(self,samples, equation_env=None, model=None, cfg_params=None, cfg=None, cond=None):
        self.samples = samples
        self.equation_env = equation_env
        self.model = model
        self.cfg=cfg
        self.cfg_params = cfg_params


        if self.cfg.prepend_conditioning_during_inference == True:
            self.state = []
            if type(cond) == dict:
                for key, value in cond.items():
                    if type(value) != torch.Tensor:
                        cond[key] = torch.tensor(value,device=self.device).unsqueeze(0)
                    else:
                        cond[key] = value.unsqueeze(0)

            elif type(cond) != torch.Tensor:
                cond = torch.tensor(cond,device=self.device).unsqueeze(0)
            else:
                cond = cond.unsqueeze(0)

            for token in cond["symbolic_conditioning"][0]:
                if "pointer" in cfg_params.id2word[token.item()]:
                    self.state.append(3)
                elif token.item() == 2:
                    self.state.append(cfg_params.word2id["partition"])
                else:
                    self.state.append(token.item())
        else:
            self.state = [cfg_params.word2id["S"]]


        self.terminal_token = cfg_params.word2id["F"]
        # state -> reward
        # we may need to retrieve the states (programs) in the order they were saved, so use OrderedDict
        self.cached_reward = OrderedDict()


    def transition(self, s, a, is_model_dynamic=True):
        if a == self.terminal_token:
            done = True
        else:
            done = False
        next_state = s + [a]
        if done:
            reward = self.get_reward(next_state)
        else:
            reward = 0 # no intermediate reward
        
        return next_state, reward, done


    def step(self, action):
        self.state, reward, done = self.transition(self.state, action)

        return self.state, reward, done, {}


    def get_reward(self, s,mode='train'):
        """
        Returns:
            The reward of program in s.
        """
        if s is None:
            return 0

        if tuple(s) in self.cached_reward.keys() and mode == 'train':
            # cache rewards for training
            return self.cached_reward[tuple(s)]


        start_time = time.time()
        _, reward, _ = compute_reward_nesymres(self.model.X ,self.model.y, s, self.cfg_params)
        print("time to get reward: ", time.time() - start_time) #bfgs for nesymres is time-consuming
         
        if mode == 'train':
            self.cached_reward[tuple(s)] = reward

        return reward

    def equality_operator(self, s1, s2):
        return s1 == s2
    
    def tokenizer_decode(self, node_action):
        return self.equation_env.equation_id2word[node_action]

    def convert_state_to_program(self, state):
        prog = []
        if type(state) != list:
            state = state.tolist()
        for i in range(len(state)):
            prog.append(self.equation_env.equation_id2word[state[i]])
        # return prog
        return " ".join(prog)