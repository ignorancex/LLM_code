import codecs
import copy
import json
import random
import re
import logging
import time

import numpy as np
from tqdm import tqdm

from src.utilities.util import sub_ge
from src.utilities.config_new import global_config as cfg
from src.simulators.multiusersimulator import MultiSimulator
from src.managers.Ross_manager import RossManager
from src.system.reward import reward
from src.dataLoader.user_dataset import get_user_act


class Interlocution(object):
    def __init__(self):
        self.error_dialog = dict()
        self.turn = 0

    def play_one_episode_for_train(self, manager:RossManager, simulator, sim_ver='rule'):
        simulator.init_state()
        manager.init_state()
        turn = 0
        self.turn += 1

        manager_act, slot_value, response = manager.generator_first_turn()
        while True:
            utterance, intent_tag, user_slot = simulator.step(response, manager_act, slot_value)

            turn += 1
            manager.maintain_state(utterance, intent_tag, user_slot)
            manager_act, slot_value, response = manager.step()

            t, s = simulator.terminal(manager_act, manager.state_mem, turn)

            if manager.trainable:
                r_sys = reward.manager_r(t, s, manager.adj)
                manager.receive_reward(r_sys)
            if simulator.trainable:
                r_usr = reward.simulator_r(t, s, simulator.adj, self.turn)
                simulator.receive_reward(r_usr)
            if t:
                break
        if manager.trainable:
            manager.register_current_episode_samples()
        if simulator.trainable:
            simulator.register_current_episode_samples()
        return 'OK'

    def play_one_episode_for_test(self, manager, simulator, turn, sim_ver='rule', save_log_path=None):
        dialog_act_seq = []
        log_dict = dict()
        turn_num = 0
        total_man_reward = 0.0
        total_sim_reward = 0.0
        slot_acc = False
        simulator.init_state()
        log_dict['goal'] = simulator.goal_value
        manager.init_state()
        manager_act, slot_value, response = manager.generator_first_turn()
        log_dict[0] = {
            'manager_action': manager.act2semantic(manager_act),
            # 'manager_slot_value': slot_value,
            'slot_memory': list(manager.state_mem),
        }
        dialog_act_seq.append(manager_act)
        
        while True:
            utterance, intent_tag, user_slot = simulator.step(response, manager_act, slot_value, test=True)

            if sim_ver == 'rule':
                log_dict[turn_num].update({
                    'simulator': utterance,
                    'simulator_slot': str(user_slot),
                    'simulator_MUM': str(simulator.slot_mem),
                    'simulator_adj': simulator.adj
                })
            else:
                log_dict[turn_num].update({
                    'simulator': utterance,
                    'nlu_act': cfg.sys_act_id2name[simulator.sys_act],
                    'nlu_slot': str(simulator.nlu_v),
                    'nlu_tsrget': str(simulator.t_nlu_v),
                    'simulator_act': cfg.user_act_id2name[intent_tag],
                    'simulator_slot': str(user_slot),
                    'simulator_adj': simulator.adj
                })
            dialog_act_seq.append(intent_tag)
            turn_num += 1

            manager.maintain_state(utterance)
            manager_act, slot_value, response = manager.step(test=True)

            log_dict[turn_num] = {
                'manager': response,
                'manager_action': manager.act2semantic(manager_act),
                'nlu': f'{cfg.user_act_id2name[manager.utterance_intent_tags]}',
                'slot_memory': str(manager.state_mem),
                'manager_adj': manager.adj,
                'state_value': round(manager.state_value, 4)
            }

            dialog_act_seq.append(manager_act)

            t, s = simulator.terminal_test(manager_act, manager.state_mem, turn_num)

            if turn_num > 20 and not s:
                t = True
                s = False

            # r_glo = simulator.global_reward(t, s)
            if manager.trainable:
                r_sys = reward.manager_r(t, s, manager.adj)
                total_man_reward += r_sys
            if simulator.trainable:
                r_usr = reward.simulator_r(t, s, simulator.adj, turn)
                total_sim_reward += r_usr

            if t or manager_act == cfg.sys_act_name2id['bye']:
                if ''.join(manager.state_mem) == ''.join(simulator.goal_value):
                    slot_acc = True
                break
        log_dict['result'] = {
            'target_value': ''.join(simulator.goal_value),
            'colted_value': ''.join(manager.state_mem)
        }
        manager.clear_samples()
        if simulator.trainable:
            simulator.clear_samples()

        if cfg.save_log and save_log_path != None:
            with codecs.open(save_log_path, 'w', 'utf-8') as f:
                json.dump(log_dict, f, ensure_ascii=False, indent=4)
        return turn_num, s, (total_man_reward, total_sim_reward), slot_acc, ' '.join(str(i) for i in dialog_act_seq)

    def test_episodes(self, manager, simulator, test_step, sim_ver='rule'):
        success_len = 0.0
        success_count = 0.0
        total_man_reward = 0.0
        total_sim_reward = 0.0
        success_episodes_seq = dict()
        sys_act_seqs = set()
        success_c_null = 0.0
        slot_success_count = 0.0
        for i in range(test_step):
            # try:
            if hasattr(simulator, 'user_group'):
                simulator.change_user()

            l, s, (man_r, sim_r), slot_acc, act_seq = self.play_one_episode_for_test(manager, simulator, i, sim_ver)
            if s:
                success_len += l
                success_count += 1
                if success_episodes_seq.get(act_seq, 0) == 0:
                    success_episodes_seq[act_seq] = 0
                success_episodes_seq[act_seq] += 1
            total_man_reward += man_r
            total_sim_reward += sim_r
            slot_success_count += slot_acc
            # except:
            #     pass

        aver_succ_len = success_len / success_count if success_count != 0 else 0.0
        # act_seq_avg_len = len(success_episodes_seq) / success_count if success_count != 0 else 0.0

        return aver_succ_len, success_count / test_step, slot_success_count / test_step, (
        total_man_reward / test_step, total_sim_reward / test_step), len(success_episodes_seq)

    @staticmethod
    def test_nlu(net, dataloader):
        intent_succ = 0.0
        slot_succ = 0.0
        count = 0

        for i, batch in enumerate(dataloader):
            utter, utter_len, slot_target, intent_target = batch
            slot_log, intent_log, _ = net(utter, utter_len)
            slot_log = slot_log[0].max(1)[1].tolist()
            intent_log = intent_log.max(1)[1].item()
            slot_target = slot_target[0].tolist()
            intent_target = intent_target.max(1)[1].item()
            if slot_log == slot_target:
                slot_succ += 1
            if intent_log == intent_target:
                intent_succ += 1
            count += 1

        return intent_succ / count, slot_succ / count, (intent_succ/count) * (slot_succ/count)

    def test_manager_dialog_success(self, manager, name, dialog):
        real_success = False
        generator_success = False
        state_slot_acc = 0
        state_joint_acc = 0
        slot_acc = 0
        state_total_num = 0
        policy_target = []
        policy_predict = []

        manager.init_state()
        goal_value = json.loads(dialog['goal_value'])
        turn = 0
        real_success = dialog['results']["target_value"] == dialog['results']['collected_value']

        first_turn = dialog['turns'][0]
        if 'staff' in first_turn:
            turn += 1
        while turn < len(dialog['turns']):
            utterance = dialog['turns'][turn]['user']
            utterance = sub_ge(utterance)
            user_act = get_user_act(dialog['turns'][turn]['user_label'], dialog['turns'][turn]['tag'])
            turn += 1
            if turn == len(dialog['turns']):
                break
            manager.maintain_state(utterance)
            manager_state = copy.deepcopy(manager.state_mem)
            staff_state = json.loads(dialog['turns'][turn]['staff_state'])
            iter_slot_acc = 0
            for i in range(len(staff_state)):
                if i < len(manager_state):
                    iter_slot_acc += manager_state[i] == staff_state[i]
                else:
                    continue
            if len(staff_state) > 0:
                state_slot_acc += iter_slot_acc / len(staff_state)
            else:
                state_slot_acc += 1 if len(manager_state) == 0 else 0

            state_joint_acc += (''.join(manager_state) == ''.join(staff_state))
            if user_act not in self.error_dialog.keys():
                self.error_dialog[user_act] = dict(s_count=0, t_count=0)
            self.error_dialog[user_act]['t_count'] += 1
            if ''.join(manager_state) == ''.join(staff_state):
                self.error_dialog[user_act]['s_count'] += 1

            state_total_num += 1
            manager_act, slot_value, _ = manager.step()
            staff_label = dialog['turns'][turn]['staff_label']
            if '(' not in staff_label:
                staff_label += '()'
            target_act = re.findall(r'(.*)[(]', staff_label)[0].lower()
            target_act = cfg.sys_act_name2id[target_act]
            slot = ''.join(s[0] for s in slot_value)
            target_slot = re.findall(r'[(](.*?)[)]', staff_label)[0]
            if target_act != 'compare':
                target_slot = target_slot.replace(',', '')
            else:
                target_slot = target_slot.split(',')
                for sl in target_slot:
                    if ''.join(staff_state).find(sl) != -1:
                        target_slot = sl
            slot_acc += target_slot == slot
            policy_predict.append(manager_act)
            policy_target.append(target_act)
            turn += 1

        if ''.join(manager.state_mem) == ''.join(goal_value):
            generator_success = True
        return real_success, generator_success, state_slot_acc / state_total_num, state_joint_acc / state_total_num, policy_predict, policy_target, slot_acc / state_total_num

    def test_gd_manager_dialog_success(self, manager, name, dialog):
        real_success = False
        generator_success = False
        state_slot_acc = 0
        state_joint_acc = 0
        slot_acc = 0
        state_total_num = 0
        policy_target = []
        policy_predict = []

        manager.init_state()
        goal_value = json.loads(dialog['goal_value'])
        turn = 0
        real_success = dialog['results']["target_value"] == dialog['results']['collected_value']

        first_turn = dialog['turns'][0]
        if 'staff' in first_turn:
            turn += 1
        while turn < len(dialog['turns']):
            utterance = dialog['turns'][turn]['user']
            utterance = sub_ge(utterance)
            turn += 1
            if turn == len(dialog['turns']):
                break
            manager.maintain_state(utterance)
            manager_state = copy.deepcopy(manager.state_mem)
            staff_state = json.loads(dialog['turns'][turn]['staff_state'])
            iter_slot_acc = 0
            for i in range(len(staff_state)):
                if i < len(manager_state):
                    iter_slot_acc += manager_state[i] == staff_state[i]
                else:
                    continue
            if len(staff_state) > 0:
                state_slot_acc += iter_slot_acc / len(staff_state)
            else:
                state_slot_acc += 1 if len(manager_state) == 0 else 0

            state_joint_acc += (''.join(manager_state) == ''.join(staff_state))
            state_total_num += 1
            manager_act, slot_value, _ = manager.step()
            staff_label = dialog['turns'][turn]['staff_label']
            if '(' not in staff_label:
                staff_label += '()'
            target_act = re.findall(r'(.*)[(]', staff_label)[0].lower()
            target_act = cfg.sys_act_name2id[target_act]
            slot = ''.join(s[0] for s in slot_value)
            target_slot = re.findall(r'[(](.*?)[)]', staff_label)[0]
            if target_act != 'compare':
                target_slot = target_slot.replace(',', '')
            else:
                target_slot = target_slot.split(',')
                for sl in target_slot:
                    if ''.join(staff_state).find(sl) != -1:
                        target_slot = sl
            slot_acc += target_slot == slot
            policy_predict.append(manager_act)
            policy_target.append(target_act)
            turn += 1

        if ''.join(manager.state_mem) == ''.join(goal_value):
            generator_success = True
        return real_success, generator_success, state_slot_acc / state_total_num, state_joint_acc / state_total_num, policy_predict, policy_target, slot_acc / state_total_num

    def save_log(self, path):
        json.dump(self.error_dialog, open(path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print('error log saved at {}.'.format(path))

    @staticmethod
    def get_diversity(manager:RossManager, simulator, epoch):
        all_path = {}
        total_time = 0
        succ_count = 0
        total_turn = 0
        for _ in tqdm(range(epoch)):
            path = ''
            turn_num = 0
            simulator.init_state()
            manager.init_state()
            manager_act, slot_value, response = manager.generator_first_turn()
            path += ',' + cfg.sys_act_id2name[manager_act]
            dialog_time = 0
            while True:
                utterance, intent_tag, user_slot = simulator.step(response, manager_act, slot_value, test=True)

                turn_num += 1
                path += ',' + cfg.user_act_id2name[intent_tag]
                start = time.time()
                manager.maintain_state(utterance)

                manager_act, slot_value, response = manager.step(test=True)

                end = time.time()
                dialog_time += (end-start)

                path += ',' + cfg.sys_act_id2name[manager_act]
                t,s = simulator.terminal_test(manager_act, manager.state_mem, turn_num)
                if t:
                    break

            if s:
                succ_count += 1
                total_turn += turn_num
                total_time += dialog_time

            if all_path.get(path, 0) == 0:
                all_path[path] = 1
            else:
                all_path[path] += 1


        succ_rate = succ_count / epoch
        count_path = len(all_path.keys())

        all_p = list(all_path.values())
        var_path = np.var(all_p)
        avg_turn = total_turn / succ_count if succ_count != 0 else 0
        avg_time = total_time / total_turn if total_turn != 0 else 0

        return succ_rate, count_path, var_path, avg_turn, avg_time, sorted(all_p, reverse=True), sum(all_p)/epoch






