import copy
import json
import logging
import os
import random
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm
from src.managers.Ross_manager import RossManager
from src.simulators.multiusersimulator import MultiSimulator
from src.simulators.weighted_rule_simulator import WeightedRuleSimulator
from src.system.interlocution import Interlocution
from src.utilities.config_new import global_config as cfg


def test_diversirt():
    manager_path = ''
    sim_path = ''

    # initialize simulators:
    sim = MultiSimulator()
    # sim = WeightedRuleSimulator(cfg.max_turn, 'all')

    if sim.trainable:
        sim.load_model(sim_path)

    # initialize manager:
    manager = RossManager(emb_dim=cfg.manager_emb_dim, char_dim=cfg.manager_char_dim, state_dim=cfg.manager_state_dim,
                          hidden_size=cfg.manager_hidden_dim, hidden_one_dim=cfg.manager_hidden_one_dim, hidden_two_dim=cfg.manager_hidden_two_dim,
                          sim_action_num=len(cfg.user_act_id2name), algo=cfg.algo, lr=cfg.manager_rl_learning_rate)
    manager.net.load_state_dict(torch.load(manager_path, map_location=torch.device(cfg.device)))

    inter = Interlocution()
    results = []
    for _ in range(3):
        results.append(inter.get_diversity(manager, sim, 1000))

    succ_rate = np.mean([r[0] for r in results])
    avg_count = np.mean([r[1] for r in results])
    avg_var = np.mean([r[2] for r in results])
    avg_turn = np.mean([r[3] for r in results])
    avg_time = np.mean([r[4] for r in results])

    print('succ rate:{:.4f}\tcount:{:.0f}\tvar:{:.4f}\tavg turn:{:.4f}\tavg time:{:.4f}'.format(succ_rate, avg_count, avg_var, avg_turn, avg_time))
    print('sim trainable:{}, manager path:{}, sim path:{}'.format(sim.trainable, manager_path, sim_path))

    # print(f'count:{count}, var: {var}, succ rate:{succ_rate}')
    #
    # plt.plot(range(count), sort_p)
    # plt.show()


if __name__ == '__main__':

    test_diversirt()

