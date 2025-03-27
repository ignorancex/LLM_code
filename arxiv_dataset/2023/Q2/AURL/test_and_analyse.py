import argparse
import copy
import json
import logging
import os
import random
import time
from collections import Counter

import numpy as np
import torch

from src.managers.Ross_manager import RossManager
from src.system.interlocution import Interlocution
from src.utilities import util
from src.dataLoader.new_dataloder import read_dialogue_from_data
from src.dataLoader.user_dataset import UserDataset, get_user_act
from src.utilities.config_new import global_config as cfg
from src.utilities.config import *
import warnings

from src.utilities.util import compute_f1


def evaluate_dialog_success(dialogs, manager, inter):
    manager.net.eval()
    total_num = 0
    success_count = 0
    state_acc = 0
    block_acc = 0
    slot_acc = 0
    policy_predict = []
    policy_target = []
    for name, dialog in dialogs.items():
        real_success, generator_success, iter_block_acc, iter_state_acc, iter_policy_predict, iter_policy_target, iter_slot_acc\
            = inter.test_manager_dialog_success(manager, name, dialog)
        total_num += real_success
        success_count += generator_success
        block_acc += iter_block_acc
        state_acc += iter_state_acc
        slot_acc += iter_slot_acc
        policy_predict.extend(iter_policy_predict)
        policy_target.extend(iter_policy_target)
    action_F1 = compute_f1(policy_predict, policy_target)
    return success_count/total_num, block_acc/len(dialogs), state_acc/len(dialogs), action_F1, slot_acc/len(dialogs)


def evaluate_simulator(all_dialog, sim, use_gt=True):
    sim.nlu.eval()
    sim.policy.eval()

    all_count = 0
    nlu_act_r = 0
    nlu_v_r = 0
    bs_r = 0
    policy_act_r = 0
    policy_act_f1 = 0
    policy_v_r = 0
    policy_target = []
    policy_predict = []

    for dialog in all_dialog.values():

        user_goal = json.loads(dialog['goal_value'])
        sim.init_state()
        sim.goal_value = user_goal

        t_bs_vector = [0] * len(user_goal)
        for idx in range(0, len(dialog['turns']) - 1, 2):
            t_sys_act, t_nlu_vector = UserDataset._get_nlu_result(user_goal, dialog['turns'][idx]['staff_label'], json.loads(dialog['turns'][idx]['staff_state']))
            sys_act, nlu_v, act_num, slot_vector, bs_vector = sim.step_test(dialog['turns'][idx]['staff'], cfg.sys_act_name2id[t_sys_act], t_bs_vector.copy(), t_nlu_vector, use_gt=use_gt)

            t_user_act = get_user_act(dialog['turns'][idx + 1]['user_label'], dialog['turns'][idx + 1]['tag'])
            if len(user_goal) == 1 and t_user_act == 'inform_multi':
                t_user_act = 'inform_normal'

            if t_user_act != 'restart' and t_sys_act == 'compare' or (t_sys_act == 'other' and 'compare' in dialog['turns'][idx - 2]['staff_label']):
                t_policy_vector = [i if i != 2 else 1 for i in t_nlu_vector]
            else:
                t_policy_vector = UserDataset._get_policy_result(user_goal, t_bs_vector,
                                                        dialog['turns'][idx + 1]['user_label'],
                                                        json.loads(dialog['turns'][idx + 1]['user_state']),
                                                        json.loads(dialog['turns'][idx]['staff_state']))
            t_bs_vector = UserDataset._update_bs(t_bs_vector, t_user_act, t_nlu_vector, t_policy_vector, json.loads(dialog['turns'][idx + 1]['user_state']))

            # _, user_act, user_slot = sim.step(dialog['turns'][idx]['staff'], cfg.sys_act_name2id[t_sys_act], t_nlu_vector, test=True)
            all_count += 1
            nlu_act_r += (t_sys_act == cfg.sys_act_id2name[sys_act])
            nlu_v_r += (t_nlu_vector[:len(user_goal)] == nlu_v[0][:len(user_goal)])
            bs_r += (t_bs_vector == bs_vector)
            policy_v_r += (t_policy_vector==slot_vector)
            policy_act_r += (t_user_act == cfg.user_act_id2name[act_num])
            policy_target.append(cfg.user_act_name2id[t_user_act])
            policy_predict.append(act_num)

    policy_act_f1 = compute_f1(policy_predict, policy_target)
    return nlu_act_r/all_count, nlu_v_r/all_count, policy_act_r/all_count, policy_act_f1, policy_v_r/all_count


def evaluate_simulator2(all_dialog, sim:MultiSimulator2, use_gt=True):
    sim.net.eval()

    all_count = 0
    nlu_act_r = 0
    nlu_v_r = 0
    policy_act_r = 0
    policy_act_f1 = 0
    policy_v_r = 0
    policy_target = []
    policy_predict = []

    for dialog in all_dialog.values():

        user_goal = json.loads(dialog['goal_value'])
        sim.init_state()
        sim.goal_value = user_goal

        t_bs_vector = [0] * len(user_goal)
        for idx in range(0, len(dialog['turns']) - 1, 2):
            t_sys_act, t_nlu_vector = UserDataset._get_nlu_result(user_goal, dialog['turns'][idx]['staff_label'], json.loads(dialog['turns'][idx]['staff_state']))
            sys_act, nlu_v, act_num, slot_vector = sim.step_test(dialog['turns'][idx]['staff'])
            sim.history += dialog['turns'][idx]['staff'] + cfg.split_token + dialog['turns'][idx+1]['user'] + cfg.split_token

            t_user_act = get_user_act(dialog['turns'][idx + 1]['user_label'], dialog['turns'][idx + 1]['tag'])
            if len(user_goal) == 1 and t_user_act == 'inform_multi':
                t_user_act = 'inform_normal'

            if t_user_act != 'restart' and t_sys_act == 'compare' or (t_sys_act == 'other' and 'compare' in dialog['turns'][idx - 2]['staff_label']):
                t_policy_vector = [i if i != 2 else 1 for i in t_nlu_vector]
            else:
                t_policy_vector = UserDataset._get_policy_result(user_goal, t_bs_vector,
                                                        dialog['turns'][idx + 1]['user_label'],
                                                        json.loads(dialog['turns'][idx + 1]['user_state']),
                                                        json.loads(dialog['turns'][idx]['staff_state']))
            t_bs_vector = UserDataset._update_bs(t_bs_vector, t_user_act, t_nlu_vector, t_policy_vector, json.loads(dialog['turns'][idx + 1]['user_state']))

            # _, user_act, user_slot = sim.step(dialog['turns'][idx]['staff'], cfg.sys_act_name2id[t_sys_act], t_nlu_vector, test=True)
            all_count += 1
            nlu_act_r += (t_sys_act == cfg.sys_act_id2name[sys_act])
            nlu_v_r += (t_nlu_vector == nlu_v)
            policy_v_r += (t_policy_vector==slot_vector)
            policy_act_r += (t_user_act == cfg.user_act_id2name[act_num])
            policy_target.append(cfg.user_act_name2id[t_user_act])
            policy_predict.append(act_num)

    policy_act_f1 = compute_f1(policy_predict, policy_target)
    return nlu_act_r/all_count, nlu_v_r/all_count, policy_act_r/all_count, policy_act_f1, policy_v_r/all_count





if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    # logging format and level setting
    # logging format and level setting
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(args.log_level)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.device != 'cpu' and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    # initialize managers
    manager = RossManager(emb_dim=args.manager_emb_dim, char_dim=args.manager_char_dim,
                          hidden_size=args.manager_hidden_dim, state_dim=args.manager_state_dim,
                          hidden_one_dim=args.manager_hidden_one_dim,
                          hidden_two_dim=args.manager_hidden_two_dim,
                          slot_num=args.slot_num, sim_action_num=len(intent_tag_names),
                          pooling=args.pooling, algo=args.algo)

    if os.path.exists(args.model_path):
        manager.net.load_state_dict(torch.load(args.model_path))
    else:
        print('model path not exists')
        exit()
    inter = Interlocution()
    test_dialogs = read_dialogue_from_data('./data/812/test_all.json')

    dialog_success, block_acc, state_acc, policy_f1 = evaluate_dialog_success(test_dialogs, manager, inter)
    inter.save_log('error.log')
    logger.info('Test result: block_acc: {:.3f}\tstaff state acc: {:.3f}\tdialog success acc: {:.3f}'
                .format(block_acc, state_acc, dialog_success))
