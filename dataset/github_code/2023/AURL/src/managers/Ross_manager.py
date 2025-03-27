import copy
import json
import random
import re
from collections import deque, namedtuple, defaultdict
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import lr_scheduler
from tqdm import tqdm

from src.managers.RuleNLG import RuleNLG
from src.networks.sys_model import Ross
from src.utilities import util
from src.utilities.config_new import global_config as cfg
from src.utilities.util import maskedNll, AutoNormalization
from src.system.replay_buffer import Replay_buffer

Transition = namedtuple(
    'Transition',
    ('encoded_hidden', 'last_act', 'state_hidden', 'act', 'slot', 'state_mem', 'reg_exp', 'reward',
     'next_encode', 'next_state', 'next_mem', 'next_reg',  'old_policy', 'long_return', 'terminal'))
Dst_state = namedtuple('Dst_state',  ('input_idx', 'last_state_idx', 'last_state_chunk', 'target_user_act', 'target_mem'))
State = namedtuple('State', ('last_act', 'last_encode', 'last_state', 'state_mem', 'reg'))


class RossManager(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        # vocabulary
        self.voc_path = self.params.get('voc_path',
                                        './src/corpus/vocab.json')
        with open(self.voc_path, 'r', encoding='utf-8') as f:
            self.voc = json.load(f)
        self.act_space = len(cfg.sys_act_id2name)
        self.regexes = cfg.RegularExpressions
        self.nlg = RuleNLG()
        self.init_net()
        self.clear_samples()
        if self.params['algo'] in ['rwb', 'a2c', 'ppo', 'rei', 'a3c']:
            self.replay_episodes_size = self.params.get('replay_episodes_size', 6)
            self.replay_buffer = deque(maxlen=self.replay_episodes_size)
        elif self.params['algo'] == 'acer':
            self.replay_episodes_size = self.params.get('replay_episodes_size', 10000)
            self.replay_buffer = deque(maxlen=self.replay_episodes_size)
        else:
            pass
        self.dst_buffer = Replay_buffer('ross_dst')
        # self.dst_buffer = Replay_buffer('sl_dst')
        self.act_prob = None
        self.trainable = True
        self.re_update = ['([0-9]{11}).?不.?[0-9]{10,}', '不.*[0-9]{10,}.*([0-9]{11})', '改成.?([0-9]{11})']
        self.reward_normal = AutoNormalization()

    def init_net(self):

        emb_dim = self.params.get('emb_dim', 100)
        hidden_size = self.params.get('hidden_size', 32)
        state_dim = self.params.get('state_dim', 32)
        hidden_one_dim = self.params.get('hidden_one_dim', 32)
        hidden_two_dim = self.params.get('hidden_two_dim', 32)
        voc_size = self.params.get('voc_size', len(self.voc))
        self.device = self.params.get('device', cfg.device)
        slot_temps = ['号-号']
        self.slot_temps = slot_temps
        gating_dict = {"none": 0, "ptr": 1}
        self.sim_action_num = self.params.get('sim_action_num', 10)
        # self.sim_action_num = 30
        self.dropout = 0.2

        learning_rate = self.params.get('lr', 1e-5)

        self.net = Ross(emb_dim, hidden_size, state_dim, hidden_one_dim, hidden_two_dim, self.act_space,
                        self.sim_action_num, self.dropout, self.voc, slot_temps, gating_dict)
        self.net.to(self.device)
        self.nlg = RuleNLG()
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.dst_optim = torch.optim.Adam(self.net.parameters(), lr=cfg.manager_sl_learning_rate)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)

    def init_state(self):
        self.state_mem = []
        self.last_state_mem = []
        self.utterance = None
        self.utterance_intent_tags = None
        self.last_act = 1
        self.last_slot = []
        self.dialog_history = ';请您报一下您的手机号;'
        state_dim = self.params.get('state_dim', 16)
        self.last_state_hidden = torch.zeros(1, 1, state_dim).tolist()

        self.confirmed_vector = []
        self.used_act = defaultdict(int)
        self.clear_samples()
        return self.last_act, self.last_slot

    def generator_first_turn(self):
        self.last_slot = []
        self.last_act = cfg.sys_act_name2id['request']
        _, _, response = self.nlg.give_response(self.last_act, self.last_slot, self.state_mem)
        self.dialog_history = response + ';'
        self.state_value = 0
        return self.last_act, self.last_slot, response

    def maintain_state(self, utterance_input, user_act=None, user_slot=None):
        if len(utterance_input) > cfg.MAX_UTTER_LEN - 1:
            utterance_input = utterance_input[:cfg.MAX_UTTER_LEN - 1]

        # prepare input
        context = self.dialog_history + utterance_input

        input_idx = [(self.voc[c] if c in self.voc else self.voc['<UNK>']) for c in context] + [self.voc['<EOS>']]
        input_tensor = torch.LongTensor([input_idx]).to(self.device)
        dialog_history_len = [len(input_idx)]
        last_sys_state, last_state_chunk = self.get_state_seq(self.last_state_mem)
        last_sys_state_len = [len(last_sys_state)]
        last_sys_state = torch.LongTensor([last_sys_state]).to(self.device)
        last_state_chunk_tensor = torch.LongTensor([last_state_chunk]).to(self.device)
        # self.net.to(self.device)
        intent_tag, _, words, encoded_hidden = self.net.nlu(input_tensor, dialog_history_len, last_sys_state, last_sys_state_len, last_state_chunk_tensor, self.slot_temps, self.device)
        intent_logs = torch.log_softmax(intent_tag, dim=-1)
        utterance_intent_tags = intent_logs.max(1)[1].item()

        for si, _ in enumerate(self.slot_temps):
            pred = np.transpose(words[si])[0]
            st = []
            for e in pred:
                if e == '<EOS>':
                    break
                elif e in '0123456789,':
                    st.append(e)
            st = ''.join(st)
            state_mem = st
        state_mem = re.sub(',+', ',', state_mem).replace('<UNK>', '')

        # if 'update' in cfg.user_act_id2name[utterance_intent_tags] and re.search('[0-9]{11}', utterance_input) is not None:
        #     for re_pattern in self.re_update:
        #         res = re.search(re_pattern, utterance_input)
        #         if res and res.group(1) != ''.join(state_mem.split(',')):
        #             state_mem = util.update_rule_dst(self.state_mem, res.group(1))
        #             break
        if user_slot is not None:
            self.target_user_act = user_act
            self.target_mem = copy.deepcopy(self.state_mem)[:3]
            if user_act == cfg.user_act_name2id['restart']:
                self.target_mem = []
            if len(user_slot) == 0:
                pass
            elif len(''.join(s[0] for s in user_slot)) == 11:
                self.target_mem = [s[0] for s in user_slot]
            else:
                if 'inform' in cfg.user_act_id2name[user_act] or 'restart' in cfg.user_act_id2name[user_act]:
                    for value, index in user_slot:
                        if index >= len(self.target_mem):
                            self.target_mem.append(value)
                    # if len(self.target_mem) > 3 or len(''.join(self.target_mem)) > 11:
                    #     print('here')
                elif 'update' in cfg.user_act_id2name[user_act]:
                    for value, index in user_slot:
                        if index < len(self.target_mem):
                            self.target_mem[index] = value
            self.state_mem = copy.deepcopy(self.target_mem)
        else:
            self.target_user_act = None
            self.target_mem = []
            self.state_mem = state_mem.split(',') if len(state_mem) > 0 else []


        self.input_idx = input_idx
        self.last_sys_state = last_sys_state.tolist()[0]
        self.last_state_chunk = last_state_chunk
        # if self.target_mem and len(''.join(self.target_mem)) < 12:
        #     self.state_mem = copy.deepcopy(self.target_mem)
        # else:
        #     if len(state_mem) > 19: state_mem = state_mem[:19]
        #     self.state_mem = state_mem.split(',') if len(state_mem) > 0 else []
        self.utterance_intent_tags = utterance_intent_tags
        self.utterance = utterance_input
        self.utt_idx = input_idx
        self.encoded_hidden = encoded_hidden.tolist()

    def get_state_seq(self, state):
        state_list_seqs = []
        list_chunk_id = []
        for i, st in enumerate(state, start=1):
            utt_seq = [self.voc[w] for w in st]
            chunk_id = [i] * (len(st))
            if i < len(state):
                utt_seq.append(self.voc[','])
                chunk_id.append(i)
            state_list_seqs.extend(utt_seq)
            list_chunk_id.extend(chunk_id)
        state_list_seqs.append(self.voc['<EOS>'])
        list_chunk_id.append(0)
        return state_list_seqs, list_chunk_id

    def get_confirm_seq(self):
        confirm_id = self.confirmed_vector
        confirm_id.append(0)
        return confirm_id

    def get_re(self, state):
        state_re = []
        for regular in self.regexes:
            regular = re.compile(regular)
            if re.match(regular, state):
                state_re.append(1)
            else:
                state_re.append(0)
        return state_re

    def prepare_input(self):
        # last act
        last_act = torch.LongTensor([[self.last_act]]).to(self.device)
        # utterance
        utterance_tensor = torch.LongTensor([self.utt_idx]).to(self.device)
        utterance_length = [len(self.utt_idx)]
        # encoded_hidden
        encoded_hidden = torch.FloatTensor(self.encoded_hidden).to(self.device)
        # last state
        last_state_hidden = self.state_to_tensor(self.last_state_hidden)
        # state_mem
        state_mem = self.mem_to_tensor(self.state_mem)
        last_sys_state, last_state_chunk = self.get_state_seq(self.last_state_mem)
        last_sys_state_len = [len(last_sys_state)]
        last_sys_state_mem = torch.LongTensor([last_sys_state]).to(self.device)
        last_state_chunk_tensor = torch.LongTensor([last_state_chunk]).to(self.device)
        self.regex = self.get_re(''.join(self.state_mem))
        regex = torch.LongTensor([self.regex]).to(self.device)
        return {
            'state_hidden': last_state_hidden,
            'utterance': utterance_tensor,
            'utter_len': utterance_length,
            'encoded_hidden': encoded_hidden,
            'last_a': last_act,
            'state_mem': state_mem,
            'last_state_mem': last_sys_state_mem,
            'last_state_len': last_sys_state_len,
            'last_state_chunk': last_state_chunk_tensor,
            'regex': regex
        }

    def one_hot_act(self, act):
        act_vector = torch.zeros(1, self.act_space).to(self.device)
        if act != None:
            act_vector = act_vector.scatter_(1, torch.tensor([[act]]).to(self.device), 1.0)
        return act_vector

    def state_to_tensor(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        return state_tensor

    def mem_to_tensor(self, state):
        state_mem = ','.join(state)
        if len(state_mem) > cfg.MAX_STATE_LEN-1:
            state_mem = state_mem[:cfg.MAX_STATE_LEN-1]
        slot_tensor = [self.voc[w] if w in self.voc else self.voc['<UNK>'] for w in state_mem] + [self.voc['<EOS>']]
        slot_tensor.extend([self.voc['<PAD>']] * (cfg.MAX_STATE_LEN - len(slot_tensor)))
        return torch.LongTensor(slot_tensor).unsqueeze(0).to(self.device)

    def encode_to_list(self, encode_tensor):
        return encode_tensor.tolist()

    def step_surprised(self, sys_act, response):
        input_tensor = self.prepare_input()
        policy, slot, state_hidden, _ = self.net.policy_predict(input_tensor['last_a'], input_tensor['encoded_hidden'], input_tensor['state_hidden'],
                                                                input_tensor['state_mem'], input_tensor['regex'], self.device)
        acts_prob = F.softmax(policy)
        self.adj = 0.0
        act_num = cfg.sys_act_name2id[sys_act]
        self.current_episode_samples.append((self.encoded_hidden, self.last_act, self.last_state_hidden,
                                             act_num, slot.tolist(), list(self.state_mem), self.regex, acts_prob.tolist(), self.adj))
        self.last_state_hidden = self.encode_to_list(state_hidden)
        self.last_act = act_num
        self.dialog_history = response + ';'

    def step(self, test=False, supervised=False):
        input_tensor = self.prepare_input()
        policy, slot, state_hidden, state_value = self.net.policy_predict(input_tensor['last_a'], input_tensor['encoded_hidden'],
                                  input_tensor['state_hidden'], input_tensor['state_mem'], input_tensor['regex'], self.device)

        acts_prob = F.softmax(policy, dim=-1)
        self.act_prob = acts_prob[0].tolist()

        if test:
            clip_prob = acts_prob.cpu().detach().numpy()[0]
            act_num = np.random.choice(self.act_space, p=clip_prob)
        else:
            m = Categorical(acts_prob[0])
            sampled_act = m.sample()
            act_num = sampled_act.item()

        slot = torch.argmax(slot, -1).tolist()[0]

        self.used_act[act_num] += 1

        # choose slot
        slot_value, flag = [], False

        if cfg.sys_act_id2name[act_num] in ['implicit_confirm', 'explicit_confirm', 'compare']:
            for idx, s in enumerate(slot):
                if idx < len(self.state_mem):
                    if s == 1:
                        slot_value.append((self.state_mem[idx], idx))
                        flag = True
                    else:
                        if flag: break
                else:
                    break

        self.adj = util.sys_adjacency_pairs(self.used_act, cfg.user_act_id2name[self.utterance_intent_tags], cfg.sys_act_id2name[act_num], slot, self.state_mem)

        # update self.encode, last_state
        self.last_state_mem = self.state_mem

        if act_num == cfg.sys_act_name2id['repeat']:
            act_num = self.last_act
            slot_value = self.last_slot
        self.last_act = act_num
        self.last_slot = slot_value

        # self.adj = 1.0
        self.current_episode_samples.append((
            self.input_idx, self.last_sys_state, self.last_state_chunk, self.last_act, self.last_state_hidden, self.encoded_hidden,
            act_num, slot, list(self.state_mem), self.regex, acts_prob.tolist(), self.target_user_act, self.get_target_state_seq(self.target_mem)))
        # update state
        self.last_state_hidden = self.encode_to_list(state_hidden)

        _, _, response = self.nlg.give_response(act_num, ''.join(s[0] for s in slot_value), self.state_mem)

        # self.dialog_history += self.utterance + ';'
        self.dialog_history = response + ';'
        self.state_value = state_value.item()
        return act_num, slot_value, response

    def receive_reward(self, r_sys):
        r_sys = self.reward_normal(r_sys)
        self.current_episode_rewards.append(r_sys)

    def clear_samples(self):
        self.current_episode_samples = []
        self.current_episode_rewards = []

    def save(self, path):
        logging.info(f"system model saved at {path}.")
        torch.save(self.net.state_dict(), path, _use_new_zipfile_serialization=False)

    def phone_rule(self):
        phone_num = ''.join(slot for slot in self.state_mem)
        if (len(phone_num)) < 11:
            return 0
        elif len(phone_num) == 11:
            return 1
        else:
            return -1

    def act2semantic(self, act_num):
        semantic = cfg.sys_act_id2name[act_num]
        if semantic == 'implicit_confirm' or semantic == 'explicit_confirm':
            semantic = semantic + '(' + ''.join(s[0] for s in self.last_slot) + ')'
        return semantic

    def pad_seq(self, seqs, max_len=None):
        seq_len = [len(seq) for seq in seqs]
        if not max_len:
            max_len = max(seq_len)
        for seq in seqs:
            seq.extend([self.voc['<PAD>']] * (max_len - len(seq)))
        return seq_len

    def pad_vector(self, vector, id=0):
        max_len = cfg.max_chunk
        # for vector in vectors:
        #     vector.extend([id] * (max_len - len(vectors)))
        # return vectors
        return vector + [id] * (max_len - len(vector))

    def get_utt_seq(self, utterance):
        utt_seq = []
        for w in utterance:
            utt_seq.append(self.voc[w] if w in self.voc else self.voc['<UNK>'])
        utt_seq.append(self.voc['<EOS>'])
        return utt_seq

    def get_target_state_seq(self, state):
        return self.get_utt_seq(','.join(state))

    def _convert_dst_batch_to_train(self, batch, device):
        batch_len = len(batch)
        input_idx = [item.input_idx for item in batch]
        input_idx_len = self.pad_seq(input_idx)
        last_state_idx = [item.last_state_idx for item in batch]
        last_state_len = self.pad_seq(last_state_idx)
        last_state_chunk = [item.last_state_chunk for item in batch]
        last_state_chunk_len = self.pad_seq(last_state_chunk)
        input_idx = torch.LongTensor(input_idx).to(device)
        last_state_idx = torch.LongTensor(last_state_idx).to(device)
        last_state_chunk = torch.LongTensor(last_state_chunk).to(device)
        target_user_act = torch.LongTensor([[item.target_user_act] for item in batch])
        target_mem = [item.target_mem for item in batch]
        target_mem_len = self.pad_seq(target_mem, max_len=cfg.MAX_STATE_LEN)
        target_mem = torch.LongTensor(target_mem).to(device)
        intent_target = torch.zeros(batch_len, self.sim_action_num).scatter_(1, target_user_act, 1).to(device)

        return input_idx, input_idx_len, last_state_idx, last_state_len, last_state_chunk, intent_target, target_mem, batch_len

    def train_dst(self, device):
        self.net.to(device)
        e_count, m_count, h_count, total_count = self.dst_buffer.get_len()
        for level in [['all'], ['middle', 'hard'], ['hard'], ['all']]:
            for l in level:
                for idx, batch in enumerate(self.dst_buffer.get_dst_batchs(batch_size=128, level=l)):
                    input_idx, input_idx_len, last_state_idx, last_state_len, last_state_chunk, intent_target, target_mem, batch_len = self._convert_dst_batch_to_train(batch, device)
                    intent_tag, all_points_outputs, _, encoded_hidden = self.net.nlu(input_idx, input_idx_len, last_state_idx, last_state_len, last_state_chunk, self.slot_temps, device)
                    intent_loss = -torch.sum(torch.log_softmax(intent_tag + 1e-21, dim=-1) * intent_target) / batch_len
                    memory_loss = maskedNll(torch.log(all_points_outputs.squeeze(0) + 1e-21), target_mem, 0)
                    loss = intent_loss + memory_loss
                    self.dst_optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                    self.dst_optim.step()
        self.dst_buffer.clear_buffer()
        logging.info('easy rate:{:.4f}\tmiddle rate:{:.4f}\thard rate:{:.4f}\ttotal count:{:.4f}'.format(e_count, m_count, h_count, total_count))
        return e_count, m_count, h_count

    def register_current_episode_samples(self):
        """register the current episode to replay buffer"""
        assert len(self.current_episode_samples) == len(self.current_episode_rewards)
        # compute returns
        one_episode_samples = []
        long_return = []
        dis_reward = 0.0
        for r in self.current_episode_rewards[::-1]:
            dis_reward = r + 0.99 * dis_reward
            long_return.insert(0, dis_reward)
        for idx in range(len(self.current_episode_samples)):
            self.dst_buffer.add_to_buffer(Dst_state(
                copy.deepcopy(self.current_episode_samples[idx][0]),  # input_idx
                copy.deepcopy(self.current_episode_samples[idx][1]),  # last_state_idx
                copy.deepcopy(self.current_episode_samples[idx][2]), # last_state_chunk
                copy.deepcopy(self.current_episode_samples[idx][11]),  # target act
                copy.deepcopy(self.current_episode_samples[idx][12]),  # target men
            ))
            if idx != len(self.current_episode_samples) - 1:
                # ('encoded_hidden', 'last_act', 'state_hidden', 'act', 'slot', 'state_mem', 'reg_exp', 'reward',
                #  'old_policy', 'long_return', 'terminal')
                one_episode_samples.append(
                    Transition(
                        copy.deepcopy(self.current_episode_samples[idx][5]),  # encoded_hidden
                        copy.deepcopy(self.current_episode_samples[idx][3]),  # last_act
                        copy.deepcopy(self.current_episode_samples[idx][4]),  # state_hidden
                        copy.deepcopy(self.current_episode_samples[idx][6]),  # act
                        copy.deepcopy(self.current_episode_samples[idx][7]),  # slot
                        copy.deepcopy(self.current_episode_samples[idx][8]),  # state_mem
                        copy.deepcopy(self.current_episode_samples[idx][9]),   # reg
                        copy.deepcopy(self.current_episode_rewards[idx]),  # reward
                        copy.deepcopy(self.current_episode_samples[idx+1][5]),  # next encoded_hidden
                        copy.deepcopy(self.current_episode_samples[idx+1][4]),  # next state
                        copy.deepcopy(self.current_episode_samples[idx+1][8]),  # next state_mem
                        copy.deepcopy(self.current_episode_samples[idx+1][9]),  # next reg
                        copy.deepcopy(self.current_episode_samples[idx][10]),  # old policy
                        copy.deepcopy(long_return[idx]),  # return
                        0.0))  # terminal
            else:
                one_episode_samples.append(
                    Transition(
                        copy.deepcopy(self.current_episode_samples[idx][5]),  # encoded_hidden
                        copy.deepcopy(self.current_episode_samples[idx][3]),  # last_act
                        copy.deepcopy(self.current_episode_samples[idx][4]),  # state_hidden
                        copy.deepcopy(self.current_episode_samples[idx][6]),  # act
                        copy.deepcopy(self.current_episode_samples[idx][7]),  # slot
                        copy.deepcopy(self.current_episode_samples[idx][8]),  # state_mem
                        copy.deepcopy(self.current_episode_samples[idx][9]),  # reg
                        copy.deepcopy(self.current_episode_rewards[idx]),  # reward
                        copy.deepcopy(self.current_episode_samples[idx][5]),  # next encoded_hidden
                        copy.deepcopy(self.current_episode_samples[idx][4]),  # next state
                        copy.deepcopy(self.current_episode_samples[idx][8]),  # next state_mem
                        copy.deepcopy(self.current_episode_samples[idx][9]),  # next reg
                        copy.deepcopy(self.current_episode_samples[idx][10]),  # old policy
                        copy.deepcopy(long_return[idx]),  # return
                        1.0))  # terminal
        self.replay_buffer.append(one_episode_samples)
        self.clear_samples()

    # actor critic
    def train_actor_critic(self, entropy_scale=0.001, gamma=0.99, device=torch.device('cpu')):
        # self.net.to(device)
        samples = []
        for one_episode_samples in self.replay_buffer:
            samples.extend(one_episode_samples)

        encoded_hidden_batch = torch.cat([torch.tensor(item.encoded_hidden) for item in samples], dim=1).to(device)
        state_mem_batch = torch.cat([self.mem_to_tensor(item.state_mem) for item in samples], dim=0).to(device)
        last_state_hidden_batch = torch.tensor([[item.state_hidden[0][0] for item in samples]]).to(device)
        last_act_batch = torch.tensor([[item.last_act] for item in samples]).to(device)
        reg_exp_batch = torch.LongTensor([item.reg_exp for item in samples]).to(device)
        act_batch = torch.LongTensor([[item.act] for item in samples]).to(device)
        slot_batch = torch.LongTensor([item.slot for item in samples]).to(device)

        # next samples
        next_samples = [State(t.act, t.next_encode, t.next_state, t.next_mem, t.next_reg) for t in samples]
        next_encoded_hidden_batch = torch.cat([torch.tensor(item.last_encode) for item in next_samples], dim=1).to(device)
        next_last_act_batch = torch.tensor([[item.last_act] for item in next_samples]).to(device)
        next_state_mem_batch = torch.cat([self.mem_to_tensor(item.state_mem) for item in next_samples], dim=0).to(device)
        next_reg_exp_batch = torch.LongTensor([item.reg for item in next_samples]).to(device)
        next_last_state_hidden_batch = torch.tensor([[item.last_state[0][0] for item in next_samples]]).to(device)

        # compute current samples
        # intent_tag, all_points_outputs, _, _, _, p, slot, value = \
        #     self.net(input_idx, input_idx_len, last_state_idx, last_state_len, last_state_chunk,
        #              state_mem_batch, last_act_batch, self.slot_temps, last_state_hidden_batch, reg_exp_batch, False, device)
        p, slot, _, value = self.net.policy_predict(last_act_batch, encoded_hidden_batch, last_state_hidden_batch, state_mem_batch, reg_exp_batch, device)
        prob = F.softmax(p, dim=-1)
        log_prob = torch.log(F.softmax(p, dim=-1)+1e-21)
        log_slot = torch.log(slot+1e-21)
        # compute next samples
        _,_,_, next_value = self.net.policy_predict(next_last_act_batch, next_encoded_hidden_batch, next_last_state_hidden_batch, next_state_mem_batch, next_reg_exp_batch, device)

        reward_batch = torch.Tensor([[item.reward] for item in samples]).to(device)
        terminal_batch = torch.Tensor([[item.terminal] for item in samples]).to(device)
        value_target = reward_batch + (1 - terminal_batch) * next_value * gamma
        # compute loss

        value_loss = F.mse_loss(value, value_target)
        advantage = (value_target - value).detach()
        policy_loss = -torch.sum(advantage * log_prob.gather(1, act_batch))
        slot_loss = -torch.sum(advantage * log_slot.gather(2, slot_batch.unsqueeze(2)).squeeze(2))
        entropy_loss = -torch.sum(prob * log_prob)
        total_loss = value_loss + (policy_loss + slot_loss - entropy_scale * entropy_loss) / len(samples)
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.optimizer.step()
        # self.net.to(torch.device('cpu'))
        return total_loss.item()

    def train_iter(self, data_batch, slot_temps, device=torch.device('cpu')):
        self.net.to(device)
        self.net.train()
        batch_loss = 0.0
        batch_size = data_batch[0]['word_idx']['usr_utter'].size(0)
        state_hidden = torch.zeros(1, batch_size, self.params.get('state_dim', 16)).to(device)
        for turn_idx, batch in enumerate(data_batch):
            dialog_history, last_sys_state, last_sys_state_len, last_state_chunk, sys_state, dialog_history_len, last_sys_act, reg_exp =\
                batch['word_idx']['dialog_history'], batch['dst_idx']['lst_sys_state'], batch['sent_len']['lst_sys_state_len'],\
                batch['dst_idx']['lst_sys_state_chunk'], batch['dst_idx']['sys_state'], \
                batch['sent_len']['dialog_history_len'], batch['prev_act_idx']['sys'], batch['dst_idx']['reg_exp']
            dialog_history = dialog_history.to(device)
            last_sys_state = last_sys_state.to(device)
            last_state_chunk = last_state_chunk.to(device)
            sys_state = sys_state.to(device)
            last_sys_act = last_sys_act.to(device)
            reg_exp = reg_exp.to(device)

            intent_tag, all_points_outputs, _, words_class_out, state_hidden, policy, slot, value =\
                self.net(dialog_history, dialog_history_len, last_sys_state, last_sys_state_len, last_state_chunk,
                         sys_state, last_sys_act, slot_temps, state_hidden, reg_exp, None, device)

            state_hidden = state_hidden.detach()

            valid_turn = batch['valid'].to(device)

            policy_target = batch['act_idx']['sys_act']
            policy_target = torch.zeros(batch_size, self.act_space).scatter_(1, policy_target, 1).to(device)
            policy_loss = -torch.sum(torch.sum(torch.log_softmax(policy+1e-21, dim=-1) * policy_target, dim=1) * valid_turn) / batch_size

            slot_target = batch['act_idx']['sys_slot_idx'].to(device)
            # slot = slot.unsqueeze(-1)

            # slot = torch.cat([1 - slot, slot], -1)
            # trg_slot_probs = slot.gather(-1, slot_target.unsqueeze(-1).long()).squeeze(-1)
            slot_loss = -torch.sum(torch.gather(torch.log(slot+1e-21), 2, slot_target.unsqueeze(2)).squeeze(2)) / batch_size
            # slot_log_prob = torch.log(trg_slot_probs)
            # slot_loss = -slot_log_prob.sum(-1, keepdim=True).sum()

            memory_target = batch['dst_idx']['sys_state'].to(device)
            memory_loss = maskedNll(torch.log(all_points_outputs.squeeze(0)+1e-21), memory_target, 0)

            intent_target = batch['act_idx']['user_act']
            intent_target = torch.zeros(batch_size, self.sim_action_num).scatter_(1, intent_target, 1).to(device)
            intent_loss = -torch.sum(torch.sum(torch.log_softmax(intent_tag+1e-21, dim=-1) * intent_target, dim=1) * valid_turn) / batch_size

            loss = policy_loss + slot_loss + intent_loss + memory_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10)
            self.optimizer.step()

            batch_loss += loss.item()
        self.net.to(torch.device('cpu'))

        return batch_loss / len(data_batch)


    def evaluate_iter(self, data_batch, slot_temps, device=torch.device('cpu')):
        batch_size = data_batch[0]['word_idx']['usr_utter'].size(0)
        all_prediction = {}
        intent_n_correct, state_n_correct, policy_n_correct, slot_n_correct = 0, 0, 0, 0
        total_num = 0

        state_hidden = torch.zeros(1, batch_size, self.params.get('state_dim', 16)).to(device)
        for turn_idx, batch in enumerate(data_batch[1:]):
            dialog_history, last_sys_state, last_sys_state_len, last_state_chunk, sys_state, dialog_history_len, last_sys_act, reg_exp = \
                batch['word_idx']['dialog_history'], batch['dst_idx']['lst_sys_state'], batch['sent_len'][
                    'lst_sys_state_len'], \
                batch['dst_idx']['lst_sys_state_chunk'], batch['dst_idx']['sys_state'], \
                batch['sent_len']['dialog_history_len'], batch['prev_act_idx']['sys'], batch['dst_idx']['reg_exp']
            dialog_history = dialog_history.to(device)
            last_sys_state = last_sys_state.to(device)
            last_state_chunk = last_state_chunk.to(device)
            sys_state = sys_state.to(device)
            last_sys_act = last_sys_act.to(device)
            reg_exp = reg_exp.to(device)

            intent_tag, all_points_outputs, words, words_class_out, state_hidden, policy, slot, value = \
                self.net(dialog_history, dialog_history_len, last_sys_state, last_sys_state_len, last_state_chunk,
                         sys_state, last_sys_act, slot_temps, state_hidden, reg_exp, False, device)

            state_hidden = state_hidden.detach()

            valid_turn = batch['valid'].to(device)
            intent_target = batch['act_idx']['user_act'].squeeze(1).to(device)
            # intent acc
            intent_n_correct += ((torch.argmax(F.softmax(intent_tag, dim=-1), dim=-1) == intent_target) * valid_turn).sum().item()

            # state acc
            for bi in range(batch_size):
                all_prediction[bi] = {
                    "last_state": batch['dst_idx']['lst_sys_state'][bi],
                    "turn_belief": batch['belief_state'][bi]
                }
                predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                for si, _ in enumerate(slot_temps):
                    pred = np.transpose(words[si])[bi]
                    st = []
                    for e in pred:
                        if e == '<EOS>': break
                        # elif e == '<SEP>': st.append(',')
                        else: st.append(e)
                    st = ''.join(st)
                    if st == "none":
                        continue
                    else:
                        predict_belief_bsz_ptr.append(slot_temps[si] + '-' + str(st))
                all_prediction[bi]["pred_bs_ptr"] = predict_belief_bsz_ptr
                if predict_belief_bsz_ptr == [batch['belief_state'][bi]] and valid_turn[bi]:
                    state_n_correct += 1

            # policy f1
            policy = torch.argmax(torch.softmax(policy, -1), -1)
            policy_target = batch['act_idx']['sys_act'].squeeze(1).to(device)
            policy_n_correct += ((policy == policy_target) * valid_turn).sum().item()

            # slot = slot.unsqueeze(-1)
            # slot = torch.cat([1 - slot, slot], -1)
            slot = torch.argmax(slot, -1)
            slot_target = batch['act_idx']['sys_slot_idx'].to(device)
            slot_n_correct += sum(torch.equal(i, j) for i, j, k in zip(slot, slot_target, valid_turn) if k)

            total_num += valid_turn.sum().item()

        return intent_n_correct / total_num, state_n_correct / total_num, policy_n_correct / total_num, slot_n_correct / total_num

    def evaluate(self, dataloader, epoch, slot_temps, device=torch.device('cpu')):
        self.net.eval()
        self.net.to(device)
        pbar = tqdm(dataloader)
        intent_acc, state_acc, policy_f1, slot_acc = 0, 0, 0, 0
        for i, data_batch in enumerate(pbar):
            intent_iter_acc, state_iter_acc, policy_iter_f1, slot_iter_acc = self.evaluate_iter(data_batch, slot_temps, device)
            intent_acc += intent_iter_acc
            state_acc += state_iter_acc
            policy_f1 += policy_iter_f1
            slot_acc += slot_iter_acc

            pbar.set_description(f'12eDev Epoch: {epoch}, Idx: {i+1}')
        return intent_acc / len(dataloader), state_acc / len(dataloader), policy_f1 / len(dataloader), slot_acc / len(dataloader)