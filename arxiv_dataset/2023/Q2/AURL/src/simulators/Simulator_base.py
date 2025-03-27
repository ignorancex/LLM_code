import copy
import json
import random
import re
import numpy as np
import torch
from src.utilities.util import AutoNormalization
from src.utilities.config_new import global_config as cfg


class Simulator_base(object):
    def __init__(self):
        self.voc = json.load(open(cfg.vocab_path, 'r', encoding='utf-8'))
        self.templates = json.load(open(cfg.multi_user_template_path, 'r', encoding='utf-8'))
        self.net_type = cfg.user_net_type
        self.algo = cfg.algo
        self.voc = json.load(open(cfg.vocab_path, 'r', encoding='utf-8'))
        self.count = 0
        self.net_type = cfg.user_net_type
        self.algo = cfg.algo
        self.reward_normal = AutoNormalization()
        if self.algo in ['rwb', 'a2c', 'ppo', 'a3c']:
            self.replay_episodes_size = cfg.user_replay_episodes_size

    def init_state(self):
        pass

    def state_to_tensor(self, state):
        if self.net_type == 'RNN' or self.net_type == 'GRU':
            state_tensor = torch.FloatTensor(state)
        else:
            state_tensor = (torch.FloatTensor(state[0]),
                            torch.FloatTensor(state[1]))
        return state_tensor

    def encode_to_list(self, encode_tensor):
        if self.net_type == 'RNN' or self.net_type == 'GRU':
            return encode_tensor.tolist()
        else:
            return (encode_tensor[0].tolist(), encode_tensor[1].tolist())

    def generate_goal(self, idx):
        prelist = ["130", "131", "132", "133", "134", "135", "136", "137", "138", "139",
                   "147", "150", "151", "152", "153", "155", "156", "157", "158", "159",
                   "186", "187", "188", "189"]
        phone = random.choice(prelist) + "".join(random.choice("0123456789") for i in range(8))
        pattern = ([3, 4, 4], [4, 4, 3], [3, 3, 5], [5, 6], [5, 4, 2], [11])
        p = np.array([0.5, 0.199, 0.199, 0.1, 0.001, 0.001])
        index = np.random.choice(len(pattern), p=p.ravel())
        pattern = pattern[index]
        goal_value = []
        pos = 0
        if idx % 6 == 0:
            p = 3 + np.random.choice(3, p=np.array([0.6, 0.35, 0.05]).ravel())
            if p == 5:
                pattern = [3, 3, 5]
                for length in pattern[:-1]:
                    goal_value.append(phone[pos: pos + length])
                    pos += length
                goal_value.append(5 * str(random.randint(0, 9)))
            elif p == 4:
                flag_ = True
                for length in pattern:
                    if flag_ and pos != 0 and length >= 4:
                        tmp = 4 * str(random.randint(0, 9))
                        p = phone[pos:pos + (length - 4)] + tmp
                        goal_value.append(p)
                        flag_ = False
                    else:
                        goal_value.append(phone[pos: pos + length])
                    pos += length
            else:
                flag_ = True
                for length in pattern:
                    if flag_ and random.random() > 0.5 and pos != 0:
                        tmp = 3 * str(random.randint(0, 9))
                        p = phone[pos:pos + (length - 3)] + tmp
                        goal_value.append(p)
                        flag_ = False
                    else:
                        goal_value.append(phone[pos: pos + length])
                    pos += length
                if flag_:
                    tmp_i = -2 if len(pattern) > 2 else -1
                    value = goal_value[tmp_i]
                    len_ = len(value)
                    tmp = 3 * str(random.randint(0, 9))
                    value = value[:(len_ - 3)] + tmp
                    goal_value[tmp_i] = value
        else:
            for length in pattern:
                goal_value.append(phone[pos: pos + length])
                pos += length
        if idx % 400 == 0:
            goal_value = [phone]
        return goal_value

    def _mask_goal(self, goal, bs_vector):
        tmp = ''
        for g, b in zip(goal, bs_vector):
            if b in [1, 2]:
                tmp += g + cfg.split_token
            else:
                tmp += cfg.mask_token * len(g) + cfg.split_token
        return tmp

    def _check_sys_slot(self, sys_slot):
        goal_len = len(self.goal_value)
        t_nlu_v = [0] * goal_len
        for slot, idx in sys_slot:
            if idx < goal_len:
                if slot == self.goal_value[idx]:
                    t_nlu_v[idx] = 1
                else:
                    t_nlu_v[idx] = 2
        return t_nlu_v

    def prepare_input(self, last_act, sys_act, nlu_v, bs_vector, last_state_encode, device=cfg.device):
        def pad_vector(vectors, pad_id):
            for vector in vectors:
                vector.extend([pad_id] * (cfg.max_chunk - len(vector)))
            return vectors

        # sys_act_tensor = torch.zeros(1, len(cfg.sys_act_id2name), dtype=torch.int64).to(device)
        # sys_act_tensor = sys_act_tensor.scatter_(1, torch.tensor([[sys_act]]).to(device), 1.0)
        last_act_tensor = torch.LongTensor([last_act]).unsqueeze(1).to(device)
        sys_act_tensor = torch.LongTensor([sys_act]).unsqueeze(1).to(device)
        nlu_v = torch.LongTensor(pad_vector(nlu_v, 3)).to(device)
        bs_vector = torch.LongTensor(pad_vector([bs_vector], 3)).to(device)
        last_hidden_state = self.state_to_tensor(last_state_encode).to(device)
        return {'last_act': last_act_tensor, 'sys_act': sys_act_tensor, 'nlu_v': nlu_v, 'bs_vector': bs_vector, 'last_hidden_state': last_hidden_state}

    def receive_reward(self, r_user):
        r_user = self.reward_normal(r_user)
        self.current_episode_rewards.append(r_user)

    def clear_samples(self):
        self.current_episode_samples = []
        self.current_episode_rewards = []

    def update_bs(self, last_bs: list, user_act: str, nlu_vector: list, policy_vector: list) -> list:
        new_bs = copy.deepcopy(last_bs)
        if user_act == 'restart':
            new_bs = policy_vector
            return new_bs
        elif 'inform' in user_act:
            for idx, v in enumerate(zip(nlu_vector[0], policy_vector)):
                n, p = v
                if n == 2 and p == 0:
                    new_bs[idx] = 2
                elif p == 1:
                    new_bs[idx] = 1

        elif 'update' in user_act:
            for idx, v in enumerate(zip(nlu_vector[0], policy_vector)):
                n, p = v
                if (n == 2 or last_bs[idx] == 2) and p == 1:
                    new_bs[idx] = 1

        return new_bs

    def find_diffone(self, value1, value2):
        flag = False
        diff = None
        pos = None
        justone = True
        for v1, v2, pos_ in zip(value1, value2, range(len(value1))):
            if v1 != v2:
                flag = True
                diff = v1
                pos = pos_
                break
        if flag:
            if value1[pos + 1:] != value2[pos:]:
                justone = False
        elif not flag and value1[:-1] == value2:
            diff = value1[-1]
            pos = len(value1) - 1
        return diff, pos, justone

    def sub_update(self, slot_value, update_value):
        if len(slot_value) == len(update_value):
            if len(update_value) == 4:
                if slot_value[:1] == update_value[:1]:
                    return slot_value[1:], update_value[1:]
                elif slot_value[-1:] == update_value[-1:]:
                    return slot_value[:-1], update_value[:-1]
            elif len(update_value) == 5:
                r_ = random.random()
                if slot_value[:2] == update_value[:2]:
                    if r_ < 0.5:
                        return slot_value[1:], update_value[1:]
                    else:
                        return slot_value[2:], update_value[2:]
                elif slot_value[-2:] == update_value[-2:]:
                    if r_ < 0.5:
                        return slot_value[:-1], update_value[:-1]
                    else:
                        return slot_value[:-2], update_value[:-2]
            elif len(update_value) > 5:
                if slot_value[4:] == update_value[4:]:
                    return slot_value[:4], update_value[:4]
                elif slot_value[:-4] == update_value[:-4]:
                    return slot_value[-4:], update_value[-4:]
                else:
                    return slot_value, update_value
        if slot_value[0] == update_value[0]:
            return slot_value[1:], update_value[1:]
        elif slot_value[-1] == update_value[-1]:
            return slot_value[:-1], update_value[:-1]
        else:
            return slot_value, update_value

    def asr_module(self, act, last_act, slot_value):

        # intent error
        if act > cfg.user_act_name2id['bye']:
            pass

        # slot_error
        elif slot_value:
            if isinstance(slot_value, str):
                value = [i for i in slot_value]
            else:
                value = [i for i in slot_value[0][0]]

            if random.random() < 0.3:
                # add new or delete one
                ind_ = random.randint(0, len(value) - 1) if random.random() < 0.7 else len(value) - 1
                num = random.randint(0, 9)
                if random.random() < 0.7:
                    value.pop(ind_)
                else:
                    value.insert(ind_, str(num))
            else:
                if len(value) == 3:
                    value[random.randint(0, len(value) - 1)] = str(random.randint(0, 9))
                elif len(value) in [4, 5]:
                    if random.random() < 0.7:
                        value[random.randint(0, len(value) - 1)] = str(random.randint(0, 9))
                    else:
                        value[random.randint(0, len(value) - 1)] = str(random.randint(0, 9))
                        value[random.randint(0, len(value) - 1)] = str(random.randint(0, 9))
                else:
                    value[random.randint(0, len(value) - 1)] = str(random.randint(0, 9))
                    value[random.randint(0, len(value) - 1)] = str(random.randint(0, 9))
                    value[random.randint(0, len(value) - 1)] = str(random.randint(0, 9))
            value = ''.join(value)
            if isinstance(slot_value, str):
                return act, value
            else:
                slot_value = [(value, slot_value[0][1])]
        return act, slot_value

    def pad_seq(self, seqs):
        seq_len = [len(seq) for seq in seqs]
        max_len = max(seq_len)
        pad_mask = [[False] * len(seq) + [True] * (max_len - len(seq)) for seq in seqs]
        for seq in seqs:
            seq.extend([self.voc['<PAD>']] * (max_len - len(seq)))
        return seqs, pad_mask

    def pad_vector(self, vectors, pad_id, mask=False):
        mask_mat = [([True] * len(it) + [False] * (cfg.max_chunk - len(it))) for it in vectors]
        for vector in vectors:
            vector.extend([pad_id] * (cfg.max_chunk - len(vector)))
        if mask:
            return vectors, mask_mat
        else:
            return vectors

    def terminal(self, manager_act, collected_slot, turn):
        terminal = False
        success = False
        if manager_act == cfg.sys_act_name2id['bye'] and ''.join(collected_slot) == ''.join(self.goal_value):
            terminal = True
            success = True
            return terminal, success
        elif turn > 20:
            terminal = True
            success = False
            return terminal, success
        else:
            return terminal, success

    def terminal_test(self, manager_act, collected_slot, turn):
        terminal = False
        success = False
        if manager_act == cfg.sys_act_name2id['bye'] and ''.join(collected_slot) == ''.join(self.goal_value):
            terminal = True
            success = True
            return terminal, success
        elif turn > 20 or manager_act == cfg.sys_act_name2id['bye']:
            terminal = True
            success = False
            return terminal, success
        else:
            return terminal, success

    def nlg(self, act, slot_value, sys_slot):
        if 'inform' not in act and 'update' not in act and 'restart' not in act:
            idx = random.randint(0, len(self.templates['user'][act]) - 1)
            sentence = self.templates['user'][act][idx]
            return sentence
            # sentence = re.sub('X', slot_value, self.templates['system']['implicit_confirm_last'][idx])
        elif 'inform' in act:
            # inform two part
            if len(slot_value) != 1:
                idx = random.randint(0, len(self.templates['user']['inform']) - 1)
                sentence = re.sub('X', ''.join(s[0] for s in slot_value), self.templates['user']['inform'][idx])
                return sentence

            if slot_value[0][1] == 0:
                t_act = 'inform_start'
            elif slot_value[0][1] == 1:
                t_act = 'inform'
            elif slot_value[0][1] == 2:
                t_act = 'inform_last'
            else:
                t_act = 'inform'
            value = slot_value[0][0]
            idx = random.randint(0, len(self.templates['user'][t_act]) - 1)
            if value == value[0] * len(value):
                sentence = re.sub('X', f'{len(value)}个{value[0]}', self.templates['user'][t_act][idx])
                return sentence

            if act == 'inform_normal':
                sentence = re.sub('X', value, self.templates['user'][t_act][idx])
                return sentence
            elif act == 'inform_multi':
                if slot_value[0][1] != 0:
                    last_value = self.goal_value[slot_value[0][1] - 1]
                    value = last_value[random.randint(1, len(last_value) - 1):] + value
                    sentence = re.sub('X', value, self.templates['user'][t_act][idx])
                    return sentence
                else:
                    sentence = re.sub('X', value, self.templates['user'][t_act][idx])
                    return sentence
            elif act == 'inform_2x':
                idx = random.randint(0, len(self.templates['user'][act][t_act]) - 1)
                sentence = re.sub('X', value, self.templates['user'][act][t_act][idx])
                return sentence
            elif act == 'inform_tone':
                value = value[:random.randint(1, len(value) - 1)] + '，' + value
                sentence = re.sub('X', value, self.templates['user'][t_act][idx])
                return sentence
            elif act == 'inform_update':
                _, fake_value = self.asr_module(0, 0, value)
                idx = random.randint(0, len(self.templates['user'][act]) - 1)
                sentence = re.sub('X', value, self.templates['user'][act][idx])
                sentence = re.sub('Y', fake_value, sentence)
                return sentence

        elif 'restart' in act:
            if len(slot_value) != 0:
                idx = random.randint(0, len(self.templates['user']['restart_inform']) - 1)
                sentence = re.sub('X', slot_value[0][0], self.templates['user']['restart_inform'][idx])
                return sentence
            else:
                idx = random.randint(0, len(self.templates['user'][act]) - 1)
                sentence = self.templates['user'][act][idx]
                return sentence

        elif 'update' in act:
            if len(slot_value) > 1:
                if random.random() < 0.1:
                    sentence = ''
                    for value, index_ in slot_value:
                        if index_ == 0:
                            action = 'update_front'
                        elif index_ == len(self.goal_value) - 1:
                            action = 'update_last'
                        else:
                            action = 'update_middle'
                        idx = random.randint(0, len(self.templates['user'][action]) - 1)
                        template = self.templates['user'][action][idx]
                        sentence_ = re.sub('X', self.goal_value[index_], template)
                        try:
                            value = sys_slot[[s[1] for s in sys_slot].index(index_)][0]
                        except:
                            value = ''
                        sentence_ = re.sub('Y', value, sentence_)
                        sentence_ = re.sub('N', str(len(value)), sentence_)
                        sentence += '，' + sentence_
                    sentence = sentence[1:]
                    return sentence
                else:
                    idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                    template = self.templates['user']['update_special'][idx]
                    all_fake_slot = ''.join(s[0] for s in sys_slot)
                    value = ''.join(self.goal_value[i] for i, v in enumerate(self.bs_vector) if v != 0)
                    sentence = re.sub('X', value, template)
                    sentence = re.sub('Y', all_fake_slot, sentence)
                    return sentence
            else:
                if len(sys_slot) == 0:
                    idx = random.randint(0, len(self.templates['user']['update_normal']) - 1)
                    template = self.templates['user']['update_normal'][idx]
                    sentence = re.sub('X', slot_value[0][0], template)
                    return sentence
                try:
                    fake_slot = sys_slot[[s[1] for s in sys_slot].index(slot_value[0][1])][0]
                except Exception as e:
                    # logging.error(f"the error is {e}. bs vector is {self.bs_vector}, sys slot is {sys_slot}, slot value is {slot_value}, user act is {act}")
                    idx = random.randint(0, len(self.templates['user']['update_normal']) - 1)
                    template = self.templates['user']['update_normal'][idx]
                    sentence = re.sub('X', slot_value[0][0], template)
                    return sentence

                value = slot_value[0][0]
                all_fake_slot = ''.join(s[0] for s in sys_slot)
                if act == 'update_special' or len(fake_slot) > len(value) + 1 or len(fake_slot) < len(value) - 1:
                    idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                    template = self.templates['user']['update_special'][idx]
                    sentence = re.sub('X', value, template)
                    sentence = re.sub('Y', fake_slot, sentence)
                    return sentence
                elif act == 'update_one' and len(fake_slot) == len(value) - 1 or len(fake_slot) == len(value) + 1:
                    if len(fake_slot) == len(value) - 1:
                        diff_value, pos, justone = self.find_diffone(value, fake_slot)
                        if justone:
                            if pos == len(value) - 1 and fake_slot == all_fake_slot[-len(fake_slot):]:
                                idx = random.randint(0, len(self.templates['user']['update_last_miss']) - 1)
                                template = self.templates['user']['update_last_miss'][idx]
                                sentence = re.sub('X', value[-1], template)
                                return sentence
                            else:
                                if random.random() < 0.5:
                                    i = 1
                                    while pos - i >= 0 and len(re.findall(value[pos - i:pos], all_fake_slot)) != 1:
                                        i += 1
                                    if pos - i < 0:
                                        idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                                        template = self.templates['user']['update_special'][idx]
                                        sentence = re.sub('X', value, template)
                                        sentence = re.sub('Y', fake_slot, sentence)
                                        return sentence
                                    else:
                                        idx = random.randint(0, len(self.templates['user']['update_after_miss']) - 1)
                                        template = self.templates['user']['update_after_miss'][idx]
                                        sentence = re.sub('X', diff_value, template)
                                        sentence = re.sub('Y', value[pos - i:pos], sentence)
                                        return sentence
                                else:
                                    i = 2
                                    while pos + i < len(value) and len(re.findall(value[pos + 1: pos + i], all_fake_slot)) != 1:
                                        i += 1
                                    if pos + i >= len(value):
                                        idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                                        template = self.templates['user']['update_special'][idx]
                                        sentence = re.sub('X', value, template)
                                        sentence = re.sub('Y', fake_slot, sentence)
                                        return sentence
                                    else:
                                        idx = random.randint(0, len(self.templates['user']['update_before_miss']) - 1)
                                        template = self.templates['user']['update_before_miss'][idx]
                                        sentence = re.sub('X', diff_value, template)
                                        sentence = re.sub('Y', value[pos + 1: pos + i], sentence)
                                        return sentence
                        else:
                            idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                            template = self.templates['user']['update_special'][idx]
                            sentence = re.sub('X', value, template)
                            sentence = re.sub('Y', fake_slot, sentence)
                            return sentence
                    else:
                        diff_value, pos, justone = self.find_diffone(fake_slot, value)
                        if justone:
                            if len(re.findall(diff_value, all_fake_slot)) == 1:
                                idx = random.randint(0, len(self.templates['user']['update_negate']) - 1)
                                template = self.templates['user']['update_negate'][idx]
                                sentence = re.sub('X', diff_value, template)
                                return sentence
                            else:
                                count = 1
                                for i in range(pos):
                                    if fake_slot[i] == diff_value:
                                        count += 1
                                count += len(re.findall(diff_value, all_fake_slot[:all_fake_slot.index(fake_slot)]))
                                idx = random.randint(0, len(self.templates['user']['update_negate_n']) - 1)
                                template = self.templates['user']['update_negate_n'][idx]
                                sentence = re.sub('X', diff_value, template)
                                sentence = re.sub('N', str(count), sentence)
                                return sentence
                        else:
                            idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                            template = self.templates['user']['update_special'][idx]
                            sentence = re.sub('X', value, template)
                            sentence = re.sub('Y', fake_slot, sentence)
                            return sentence
                elif act == 'update_sure':
                    if slot_value[0][1] == 0:
                        action = 'update_front'
                    elif slot_value[0][1] == len(self.goal_value) - 1:
                        action = 'update_last'
                    else:
                        action = 'update_middle'
                    idx = random.randint(0, len(self.templates['user'][action]) - 1)
                    template = self.templates['user'][action][idx]
                    sentence = re.sub('X', value, template)
                    sentence = re.sub('Y', fake_slot, sentence)
                    sentence = re.sub('N', str(len(fake_slot)), sentence)
                    return sentence
                elif act == 'update_sub':
                    fake_slot_value, value = self.sub_update(fake_slot, value)
                    idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                    template = self.templates['user']['update_special'][idx]
                    sentence = re.sub('X', value, template)
                    sentence = re.sub('Y', fake_slot_value, sentence)
                    return sentence
                elif act == 'update_normal':
                    idx = random.randint(0, len(self.templates['user']['update_normal']) - 1)
                    template = self.templates['user']['update_normal'][idx]
                    sentence = re.sub('X', value, template)
                    return sentence
                else:
                    idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                    template = self.templates['user']['update_special'][idx]
                    sentence = re.sub('X', value, template)
                    sentence = re.sub('Y', fake_slot, sentence)
                    return sentence
