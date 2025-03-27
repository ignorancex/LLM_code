import copy
import json
import re
import torch
import random
from collections import namedtuple
from torch.utils.data.dataloader import Dataset, DataLoader
from src.utilities.config_new import global_config as cfg

Turn_data = namedtuple('Turn_data', 'goal, sys_uttr, sys_act, nlu_vector, last_act, bs_vector, user_act, policy_vector')


def data_loader_policy(type, batch_size=cfg.batch_size):
    if type == 'train':
        user_dataset = UserDataset(cfg.train_path)
    elif type == 'dev':
        user_dataset = UserDataset(cfg.dev_path)
    else:
        user_dataset = UserDataset(cfg.test_path)
    user_dataloader = DataLoader(user_dataset, batch_size=batch_size, collate_fn=user_dataset.collate_fn, shuffle=True)
    return user_dataloader


def get_user_act(user_label, tag):
    act_map = {
        "inform-multi_inform": "inform_multi",
        "how_signal-signal": "how_signal",
        "good_signal-signal": "good_signal",
        "restart-restart": "restart",
        "affirm-normal": "affirm",
        "inform-2X": "inform_2x",
        "inform-compare": "update_normal",
        "update-negate_n": "update_one",
        "update-update_jige": "update_normal",
        "update-negate": "update_one",
        "deny-normal": "deny",
        "inform-asr_num_to_han": "inform_normal",
        "inform-inform_jige": "inform_normal",
        "update-update_sure": "update_sure",
        "bad_signal-signal": "bad_signal",
        "update-sub_update": "update_sub",
        "update-update_two_part": "update_special",
        "ask_repeat-normal": "ask_repeat",
        "update-update_special": "update_special",
        "wait-normal": "wait",
        "ack-normal": "ack",
        "update-before_miss": "update_one",
        "update-after_miss": "update_one",
        "update-compare": "update_normal",
        "doubt_identity-robot": "doubt_identity",
        "inform-tone_inform": "inform_tone",
        "inform-inform_update": "inform_update",
        "update-last_miss": "update_one",
        "inform-inform": "inform_normal",
        "ask_state-normal": "ask_state",
        "other-jushi": "other",
        "finish-normal": "finish",
        "update-update_normal": "update_normal",
        "offer-normal": "offer"
    }
    if '(' in user_label: user_label = user_label[:user_label.rfind('(')]
    idx = user_label + '-' + tag

    if idx not in act_map.keys():
        raise Exception('Error, unknown act and tag map!')
    return act_map[idx]


class UserDataset(Dataset):
    def __init__(self, file_path,):
        super(UserDataset, self).__init__()
        self.vocab = json.load(open(cfg.vocab_path, 'r', encoding='utf-8'))
        self.mask_token = cfg.mask_token
        self.split_token = cfg.split_token
        self.max_chunk = cfg.max_chunk
        self.data = self._read_json(file_path)

    def __getitem__(self, item):
        data = self.data[item]
        return data

    def __len__(self):
        return len(self.data)

    def get_utt_seq(self, utterance):
        # utt_seq = [self.vocab['<CLS>'], self.vocab['<Slot1>'], self.vocab['<Slot2>'], self.vocab['<Slot3>']]
        utt_seq = [self.vocab['<CLS>']]
        for w in utterance:
            utt_seq.append(self.vocab[w] if w in self.vocab else self.vocab['<UNK>'])
        utt_seq.append(self.vocab['<EOS>'])
        return utt_seq

    def pad_vector(self, vectors, pad_id):
        mask_mat = [([True] * len(it) + [False] * (cfg.max_chunk - len(it))) for it in vectors]
        for vector in vectors:
            vector.extend([pad_id] * (self.max_chunk - len(vector)))
        return vectors, mask_mat

    def pad_seq(self, seqs, max_len=None):
        seq_len = [len(seq) for seq in seqs]
        if not max_len:
            max_len = max(seq_len)
        pad_mask = [[False]*len(seq)+[True]*(max_len-len(seq)) for seq in seqs]
        for seq in seqs:
            seq.extend([self.vocab['<PAD>']] * (max_len - len(seq)))
        return max_len, seqs, pad_mask

    def collate_fn(self, data_batch):
        batch_size = len(data_batch)

        # check max turns within considered dialogues
        dial_len = [dial['dial_len'] for dial in data_batch]
        max_dial_len = max(dial_len)

        batch_list = []
        for turn_idx in range(max_dial_len):
            last_act, sys_act, sys_slot, bs_vector, policy_act, policy_slot, policy_mask, valid_turn = [], [], [], [], [], [], [], []
            for dial_idx in range(batch_size):
                dial = data_batch[dial_idx]
                valid = True if turn_idx < dial['dial_len'] else False
                last_act.append([cfg.user_act_name2id[dial['last_act'][turn_idx]]] if valid else [cfg.user_act_name2id['bye']])
                sys_act.append([cfg.sys_act_name2id[dial['sys_act'][turn_idx]]] if valid else [cfg.sys_act_name2id['bye']])
                sys_slot.append(dial['sys_slot'][turn_idx] if valid else [0] * cfg.max_chunk)
                bs_vector.append(dial['bs_vector'][turn_idx] if valid else [1] * cfg.max_chunk)
                policy_act.append([cfg.user_act_name2id[dial['policy_act'][turn_idx]]] if valid else [cfg.user_act_name2id['bye']])
                policy_slot.append(dial['policy_slot'][turn_idx] if valid else [0] * cfg.max_chunk)
                policy_mask.append([True] * len(dial['policy_slot'][turn_idx]) + [False] * (cfg.max_chunk-len(dial['policy_slot'][turn_idx])) if valid else [False] * cfg.max_chunk)
                valid_turn.append(valid)

            sys_slot, _ = self.pad_vector(sys_slot, 3)
            bs_vector, _ = self.pad_vector(bs_vector, 3)
            policy_slot, _ = self.pad_vector(policy_slot, 0)

            batch = {}
            batch['last_act'] = torch.LongTensor(last_act)
            batch['sys_act'] = torch.LongTensor(sys_act)
            batch['sys_slot'] = torch.LongTensor(sys_slot)
            batch['bs_vector'] = torch.LongTensor(bs_vector)
            batch['policy_act'] = torch.LongTensor(policy_act)
            batch['policy_slot'] = torch.LongTensor(policy_slot)
            batch['policy_mask'] = torch.BoolTensor(policy_mask)
            batch['valid'] = torch.Tensor(valid_turn)

            batch_list.append(batch)

        return batch_list

    def _mask_goal(self, goal, bs_vector):
        tmp = ''
        for g, b in zip(goal, bs_vector):
            if b in [1, 2]:
                tmp += g + self.split_token
            else:
                tmp += self.mask_token * len(g) + self.split_token
        return tmp

    @staticmethod
    def _get_nlu_result(goal: list, staff_label: str, staff_state: list):
        nlu_vector = [0] * len(goal)
        if '(' not in staff_label:
            return staff_label, nlu_vector
        sys_act = staff_label[:staff_label.rfind('(')]
        sys_slot = re.findall(r'[(](.*?)[)]', staff_label)[0]
        if 'confirm' in sys_act:
            for sub_slot in sys_slot.split(','):
                try:
                    idx = staff_state.index(sub_slot)
                except:
                    raise Exception(f'cannot find sub slot, state: {staff_state}, label: {staff_label}')
                if sub_slot == goal[idx]:
                    nlu_vector[idx] = 1
                else:
                    nlu_vector[idx] = 2
        elif sys_act == 'compare':
            sub_slot = sys_slot.split(',')
            if len(goal) == 1:
                nlu_vector = [2]
            elif len(staff_state) == 2:
                nlu_vector = [2, 2, 0]
            else:
                p01 = staff_state[0] + staff_state[1]
                p12 = staff_state[1] + staff_state[2]
                if p01 in sub_slot:
                    nlu_vector = [2, 2, 0]
                elif p12 in sub_slot:
                    nlu_vector = [0, 2, 2]
                else:
                    raise Exception(f'can not deal compare case, state: {staff_state}, label: {staff_label}')
        else:
            raise Exception(f'forget deal some system action act when load data:{sys_act}')
        return sys_act, nlu_vector

    @staticmethod
    def _get_policy_result(goal: list, bs_vector: list, user_label: str, user_state: list, staff_state: list):
        policy_vector = [0] * len(goal)
        if '(' not in user_label and user_label != 'restart':
            return policy_vector
        user_act = user_label if '(' not in user_label else user_label[:user_label.rfind('(')]
        # user_slot = re.findall(r'[(](.*?)[)]', user_label)[0]
        if user_act == 'inform':
            len_informed = sum(i > 0 for i in bs_vector)
            assert len_informed == len(user_state) - 1
            policy_vector[len_informed] = 1
        elif user_act == 'update':
            for idx, v in enumerate(zip(staff_state, user_state)):
                s, u = v
                if s != u:
                    policy_vector[idx] = 1
        elif user_act == 'restart':
            if len(user_state) != 0:
                policy_vector[0] = 1
        else:
            raise Exception(f'forget deal some user action act when load data:{user_act}')
        return policy_vector

    @staticmethod
    def _update_bs(last_bs: list, user_act: str, nlu_vector: list, policy_vector: list, user_state: list) -> list:
        new_bs = copy.deepcopy(last_bs)
        if user_act == 'restart':
            new_bs = policy_vector
            return new_bs

        if len(user_state) == 0:
            new_bs = [0] * len(last_bs)
            return new_bs

        for idx, v in enumerate(zip(nlu_vector, policy_vector)):
            n, p = v
            if n == 2 and p == 0:
                new_bs[idx] = 2
            elif p == 1:
                new_bs[idx] = 1

        assert sum(i > 0 for i in new_bs) <= len(user_state)
        return new_bs

    def _read_json(self, filename):
        all_data = []
        all_dialog = json.load(open(filename, 'r', encoding='utf-8'))
        for dialog in all_dialog.values():
            dialogue = {
                "last_act": [],
                "sys_act": [],
                "sys_slot": [],
                "bs_vector": [],
                "policy_act": [],
                "policy_slot": []
            }

            user_goal = json.loads(dialog['goal_value'])
            bs_vector = [0] * len(user_goal)
            last_act = 'ack'
            for idx in range(0, len(dialog['turns']) - 1, 2):
                sys_act, nlu_vector = self._get_nlu_result(user_goal, dialog['turns'][idx]['staff_label'],
                                                           json.loads(dialog['turns'][idx]['staff_state']))
                user_act = get_user_act(dialog['turns'][idx + 1]['user_label'], dialog['turns'][idx + 1]['tag'])

                if len(user_goal) == 1 and user_act == 'inform_multi':
                    user_act = 'inform_normal'

                if user_act != 'restart' and sys_act == 'compare' or (sys_act == 'other' and 'compare' in dialog['turns'][idx - 2]['staff_label']):
                    policy_vector = [i if i != 2 else 1 for i in nlu_vector]
                else:
                    policy_vector = self._get_policy_result(user_goal, bs_vector,
                                                            dialog['turns'][idx + 1]['user_label'],
                                                            json.loads(dialog['turns'][idx + 1]['user_state']),
                                                            json.loads(dialog['turns'][idx]['staff_state']))
                dialogue['last_act'].append(last_act)
                dialogue['sys_act'].append(sys_act)
                dialogue['sys_slot'].append(nlu_vector)
                dialogue['bs_vector'].append(bs_vector)
                dialogue['policy_act'].append(user_act)
                dialogue['policy_slot'].append(policy_vector)

                bs_vector = self._update_bs(bs_vector, user_act, nlu_vector, policy_vector, json.loads(dialog['turns'][idx + 1]['user_state']))
                last_act = user_act

            dialogue['dial_len'] = len(dialogue['last_act'])
            all_data.append(dialogue)
        return all_data


if __name__ == '__main__':
    random.seed(9)
    dataloader = data_loader('train', 4)
    for idx, i in enumerate(dataloader):
        print(idx)
        if idx == 5474:
            print('here')
        pass
