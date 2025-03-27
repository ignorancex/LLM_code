'''
1. Modify the sb3 dummy_vec_env to deal with multiple rewards
original code: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/dummy_vec_env.py

2. Modify the sb3 Monitor to deal with multiple rewards
original code: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/monitor.py

3. evaluation
evaluate the model during training (instead of saving checkpoints as done in APPO's code)
'''
import time

from stable_baselines3.common.vec_env.dummy_vec_env import *
from typing import Dict, List, Tuple

from stable_baselines3.common.monitor import * 
from stable_baselines3.common.type_aliases import GymObs

import numpy as np
import torch
import random
import copy

class DummyVecEnv_fair(DummyVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        super().__init__(env_fns)
        
        self.num_groups = env_fns[0]().env.num_groups
        self.buf_rews = [np.zeros((self.num_envs,), dtype=np.float32),[np.zeros((self.num_envs,), dtype=np.float32) for g in range(self.num_groups)],[np.zeros((self.num_envs,), dtype=np.float32) for g in range(self.num_groups)]]

    def step_wait(self) -> Tuple[VecEnvObs, List[np.ndarray], np.ndarray, List[Dict]]:
        for env_idx in range(self.num_envs):
            obs, buf_rews_list_env_idx, self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            self.buf_rews[0][env_idx] = buf_rews_list_env_idx[0]
            for g in range(self.num_groups):
                self.buf_rews[1][g][env_idx] = buf_rews_list_env_idx[1][g]
                self.buf_rews[2][g][env_idx] = buf_rews_list_env_idx[2][g]

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), copy.deepcopy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))
    
class Monitor_fair(Monitor):
    def __init__(self, env: gym.Env, filename: Optional[str] = None, allow_early_resets: bool = True, reset_keywords: Tuple[str, ...] = (), info_keywords: Tuple[str, ...] = ()):
        super().__init__(env, filename, allow_early_resets, reset_keywords, info_keywords)

        self.num_groups = env.num_groups
        # e.g.: [ [], [[],[],[]], [[],[],[]] ]
        self.rewards: List[Union[List[float],List[List[float]]]] = [[],[[] for g in range(self.num_groups)], [[] for g in range(self.num_groups)]] 
        self.episode_returns: List[Union[List[float],List[List[float]]]] = [[],[[] for g in range(self.num_groups)], [[] for g in range(self.num_groups)]]
    
    def reset(self, **kwargs) -> GymObs:
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = [[],[[] for g in range(self.num_groups)], [[] for g in range(self.num_groups)]] 
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(f"Expected you to pass keyword argument {key} into reset")
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)
    
    def step(self, action: Union[np.ndarray, int]) -> Tuple[GymObs, List[float], bool, Dict]:
        """
        Step the environment with the given action
        :param action: the action
        :return: observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)

        self.rewards[0].append(reward[0])
        for g in range(self.num_groups):
            self.rewards[1][g].append(reward[1][g])
            self.rewards[2][g].append(reward[2][g])

        if done:
            self.needs_reset = True
            ep_rew = [sum(self.rewards[0]), [sum(self.rewards[1][g]) for g in range(self.num_groups)], [sum(self.rewards[2][g]) for g in range(self.num_groups)]]
            ep_len = len(self.rewards[0])
            assert ep_len == len(self.rewards[1][0]), 'reward lengths are different'
            ep_info = {"r": round(ep_rew[0], 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]

            self.episode_returns[0].append(ep_rew[0])
            for g in range(self.num_groups):
                self.episode_returns[1][g].append(ep_rew[1][g])
                self.episode_returns[2][g].append(ep_rew[2][g])
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info
    
    def get_episode_rewards(self) -> List[List[float]]:
        """
        Returns the rewards of all the episodes
        :return:
        """
        return self.episode_returns

def evaluate_fair(env, agent, num_eps):
    '''
    A general function to evaluate the reward and fairness of a policy
    env: should be the one with fairness reward signals (supply and demand)
    num_eps: number of episodes

    Note that bias is computed using the supply and demand of the env, not using specific env's states (such as "incident_seen")

    return:
    1. reward
    2. bias
    '''
    assert str('ActorCriticPolicy_fair') in str(type(agent)), 'evaluate_fair only works for ActorCriticPolicy_fair policy'
    assert str('PPOEnvWrapper_fair') in str(type(env)), 'env should be of type: PPOEnvWrapper_fair and should not be vectorized here'

    num_groups = env.num_groups
    seeds = [random.randint(0, 10000) for _ in range(num_eps)]
    num_timesteps = env.ep_timesteps # number of steps per episodes (unless done=True) 

    agent.set_training_mode(False)

    rewards_all = np.zeros((num_eps, num_timesteps))
    U_all = np.zeros((num_eps, num_groups, num_timesteps))
    B_all = np.zeros((num_eps, num_groups, num_timesteps))


    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])

        obs = env.reset()
        done = False
        
        for t in range(num_timesteps):
            with torch.no_grad():
                action = agent.predict(obs)[0]

            obs, r, done, _ = env.step(action) # reward is a "Fairness List"

            rewards_all[ep][t] = r[0]
            for g in range(num_groups):
                U_all[ep][g][t] = r[1][g]
                B_all[ep][g][t] = r[2][g]

            if done:
                break

    U = np.sum(U_all,axis=(0,2))
    B = np.sum(B_all,axis=(0,2)) + 1 * num_eps # 1 * num_eps is according to the formula in APPO's paper

    # essential (only write these to disk): average across episodes and timesteps
    eval_data_essential = {}
    eval_data_essential['return'] = rewards_all.mean() # average across episodes and timesteps
    ratio_list = []
    for g in range(num_groups):
        eval_data_essential['ratio_{}'.format(g)] = U[g]/B[g]
        eval_data_essential['demand_{}'.format(g)] = B[g]/(num_eps*num_timesteps)
        eval_data_essential['supply_{}'.format(g)] = U[g]/(num_eps*num_timesteps)
        ratio_list.append(U[g]/B[g])
    eval_data_essential['bias'] = max(ratio_list) - min(ratio_list)
    max_group = np.argmax(ratio_list)
    min_group = np.argmin(ratio_list)
    eval_data_essential['benefit_max'] = max(ratio_list)
    eval_data_essential['benefit_min'] = min(ratio_list)
    eval_data_essential['demand_max'] = B[max_group]/(num_eps*num_timesteps)
    eval_data_essential['demand_min'] = B[min_group]/(num_eps*num_timesteps)
    eval_data_essential['supply_max'] = U[max_group]/(num_eps*num_timesteps)
    eval_data_essential['supply_min'] = U[min_group]/(num_eps*num_timesteps)

    return eval_data_essential