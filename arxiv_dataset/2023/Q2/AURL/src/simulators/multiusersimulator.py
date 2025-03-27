import copy
import json
import random
import re
import logging
from collections import namedtuple, deque
import os
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from src.utilities import util
from src.utilities.config_new import global_config as cfg
from src.networks.user_model import UtteranceEncoder, Policy
from sklearn.metrics import f1_score
from src.dataLoader.user_dataset import get_user_act, UserDataset
from src.simulators.Simulator_base import Simulator_base

if cfg.device != 'cpu' and torch.cuda.is_available(): torch.cuda.set_device(int(cfg.device[-1]))

Transition = namedtuple('Transition', ('sys_uttr', 't_sys_act', 't_nlu_v', 'last_act', 'sys_act', 'bs_vector', 'nlu_v', 'state_encode', 'user_act', 'user_slot', 'goal_len', 'reward',
                                       'next_sys_act', 'next_bs_vector', 'next_nlu_v', 'next_state_encode', 'old_policy', 'old_slot_policy', 'long_return', 'terminal'))

State = namedtuple(
    'State', ('last_act', 'sys_act', 'bs_vector', 'nlu_v', 'state_encode'))

class MultiSimulator(Simulator_base):
    def __init__(self, train='rl'):
        super(MultiSimulator, self).__init__()
        self.nlu = UtteranceEncoder().to(cfg.device)
        self.policy = Policy().to(cfg.device)
        self.init_optimizer(train)
        self.replay_buffer = deque(maxlen=self.replay_episodes_size)
        self.trainable = True
        self.init_state()


    def init_state(self):
        self.count += 1
        self.last_act = cfg.user_act_name2id['other']
        self.goal_value = self.generate_goal(self.count)
        self.bs_vector = [0] * len(self.goal_value)
        self.clear_samples()
        self.turn = 0
        if self.net_type == 'RNN' or self.net_type == 'GRU':
            self.last_state_encode = torch.zeros(1, 1, cfg.user_state_hidden_dim).tolist()
        else:
            self.last_state_encode = (torch.zeros(1, 1, cfg.user_state_hidden_dim).tolist(),
                                      torch.zeros(1, 1, cfg.user_state_hidden_dim).tolist())

    def init_optimizer(self, learn_type='sl'):
        lr = cfg.user_sl_lr if learn_type == 'sl' else cfg.user_rl_lr
        self.nlu_optim = torch.optim.Adam(self.nlu.parameters(), lr=cfg.user_sl_lr)
        self.policy_optim = torch.optim.RMSprop(self.policy.parameters(), lr=lr)

    def maintain_state(self, uttr, t_sys_act, sys_slot, device=cfg.device):
        uttr = self._mask_goal(self.goal_value, self.bs_vector) + cfg.split_token + uttr
        uttr = torch.LongTensor([[self.voc['<CLS>'], self.voc['<Slot1>'], self.voc['<Slot2>'], self.voc['<Slot3>']] + [self.voc[w] if w in self.voc else self.voc['<UNK>'] for w in uttr] + [self.voc['<EOS>']]]).to(device)
        sys_act, nlu_v, _ = self.nlu(uttr.transpose(0, 1))
        sys_act = torch.argmax(sys_act).item()
        nlu_v = torch.argmax(nlu_v, dim=-1).tolist()

        self.sys_uttr = uttr
        self.sys_act = sys_act
        self.nlu_v = nlu_v
        self.t_sys_act = t_sys_act
        self.t_nlu_v = self._check_sys_slot(sys_slot)

    def step_test(self, uttr, t_sys_act, bs_vector, t_sys_slot, device=cfg.device, use_gt=True):
        uttr = self._mask_goal(self.goal_value, bs_vector) + cfg.split_token + uttr
        uttr = torch.LongTensor(
            [[self.voc['<CLS>'], self.voc['<Slot1>'], self.voc['<Slot2>'], self.voc['<Slot3>']] + [self.voc[w] if w in self.voc else self.voc['<UNK>'] for w in uttr] + [self.voc['<EOS>']]]).to(
            device)
        sys_act, nlu_v, _ = self.nlu(uttr.transpose(0, 1))
        sys_act = torch.argmax(sys_act).item()
        nlu_v = torch.argmax(nlu_v, dim=-1).tolist()
        # if use_gt:
        #     net_input = self.prepare_input(t_sys_act, [t_sys_slot], bs_vector, device)
        # else:
        net_input = self.prepare_input(self.last_act, sys_act, nlu_v, bs_vector, self.last_state_encode, device)
        p_user_act, p_user_slot, _, _, encode_tensor = self.policy(net_input['last_act'], net_input['sys_act'], net_input['bs_vector'], net_input['nlu_v'], net_input['last_hidden_state'])
        self.last_state_encode = self.encode_to_list(encode_tensor)
        clip_prob = p_user_act.detach().cpu().numpy()[0]
        act_num = np.random.choice(len(cfg.user_act_id2name), p=clip_prob)
        self.last_act = act_num
        slot_vector = [np.random.choice([0, 1], p=i) for i in p_user_slot.detach().cpu().numpy()[0]]
        bs_vector = self.update_bs(bs_vector, cfg.user_act_id2name[act_num], [t_sys_slot], slot_vector)
        return sys_act, nlu_v, act_num, slot_vector, bs_vector

    def step(self, uttr, t_sys_act, sys_slot, test=False, device=cfg.device):
        self.nlu.eval()
        self.policy.eval()
        if not test: self.turn += 1
        self.maintain_state(uttr, t_sys_act, sys_slot, device)
        net_input = self.prepare_input(self.last_act, self.sys_act, self.nlu_v, self.bs_vector, self.last_state_encode, device)
        p_user_act, p_user_slot, _, _, encode_tensor = self.policy(net_input['last_act'], net_input['sys_act'], net_input['bs_vector'], net_input['nlu_v'], net_input['last_hidden_state'])

        if test:
            clip_prob = p_user_act.detach().cpu().numpy()[0]
            act_num = np.random.choice(len(cfg.user_act_id2name), p=clip_prob)
            slot_vector = torch.argmax(p_user_slot, dim=-1).tolist()[0]
            # slot_vector = [np.random.choice([0, 1], p=i) for i in p_user_slot.detach().cpu().numpy()[0]]
        else:
            m = Categorical(p_user_act[0])
            sampled_act = m.sample()
            act_num = sampled_act.item()
            slot_vector = torch.argmax(p_user_slot, dim=-1).tolist()[0]

        slot_value, flag = [], False
        for idx, s in enumerate(slot_vector):
            if idx >= len(self.goal_value):
                break
            if s == 1:
                flag = True
                slot_value.append((self.goal_value[idx], idx))
            else:
                if 'update' not in cfg.user_act_id2name[act_num] and flag:
                    break

        if sum(slot_vector) <= 1 and len(slot_value) != 0 and 'inform' in cfg.user_act_id2name[act_num] and random.random() < cfg.user_error_prob:
            act_num, slot_value = self.asr_module(act_num, cfg.user_act_id2name[self.last_act], slot_value)

        try:
            response = self.nlg(cfg.user_act_id2name[act_num], slot_value, sys_slot)
        except:
            act_num = cfg.user_act_name2id['ask_repeat']
            slot_value = []
            response = self.nlg(cfg.user_act_id2name[act_num], slot_value, sys_slot)

        self.adj = util.user_adjacency_pairs(self.sys_act, act_num, slot_vector, self.bs_vector, self.t_nlu_v)
        self.current_episode_samples.append((self.sys_uttr, self.t_sys_act, self.t_nlu_v, self.last_act, self.sys_act, net_input['bs_vector'], net_input['nlu_v'],
                                             act_num, slot_vector, p_user_act.tolist(), p_user_slot.tolist(), self.adj, self.last_state_encode))
        self.last_state_encode = self.encode_to_list(encode_tensor)
        self.last_act = act_num
        self.bs_vector = self.update_bs(self.bs_vector, cfg.user_act_id2name[act_num], self.nlu_v, slot_vector)
        return response, act_num, slot_value

    def register_current_episode_samples(self):
        """register the current episode to replay buffer"""
        assert len(self.current_episode_samples) == len(
            self.current_episode_rewards)
        one_episode_samples = []
        long_return = []
        dis_reward = 0.0
        for r in self.current_episode_rewards[::-1]:
            dis_reward = r + 0.99 * dis_reward
            long_return.insert(0, dis_reward)

        for idx in range(len(self.current_episode_samples)):
            if idx != len(self.current_episode_samples) - 1:
                # Transition = namedtuple('Transition',
                #    ('sys_uttr', 't_sys_sct', 't__nlu_v', 'sys_act', 'bs_vector', 'nlu_v', 'state_encode', 'user_act', 'user_slot', 'reward', 'old_policy', 'old_slot_policy', 'return', 'terminal'))
                one_episode_samples.append(
                    Transition(copy.deepcopy(self.current_episode_samples[idx][0]),  # sys uttr
                               copy.deepcopy(self.current_episode_samples[idx][1]),  # t_sys_act
                               copy.deepcopy(self.current_episode_samples[idx][2]),  # t_nlu_v
                               copy.deepcopy(self.current_episode_samples[idx][3]),  # last_act
                               copy.deepcopy(self.current_episode_samples[idx][4]),  # sys_act
                               copy.deepcopy(self.current_episode_samples[idx][5]),  # bs vector
                               copy.deepcopy(self.current_episode_samples[idx][6]),  # sys_slot
                               copy.deepcopy(self.current_episode_samples[idx][12]),    # encode
                               copy.deepcopy(self.current_episode_samples[idx][7]),  # current act
                               copy.deepcopy(self.current_episode_samples[idx][8]),  # current slot
                               copy.deepcopy(len(self.goal_value)),  # goal len
                               copy.deepcopy(self.current_episode_rewards[idx]),  # reward
                               copy.deepcopy(self.current_episode_samples[idx + 1][4]),  # next manager act
                               copy.deepcopy(self.current_episode_samples[idx + 1][5]),  # next bs
                               copy.deepcopy(self.current_episode_samples[idx + 1][6]),  # next nlu v
                               copy.deepcopy(self.current_episode_samples[idx + 1][12]),  # next encode
                               copy.deepcopy(self.current_episode_samples[idx][9]),  # old policy
                               copy.deepcopy(self.current_episode_samples[idx][10]),  # old slot policy
                    copy.deepcopy(long_return[idx]),
                    0.0))

            else:
                one_episode_samples.append(
                    Transition(copy.deepcopy(self.current_episode_samples[idx][0]),  # sys uttr
                               copy.deepcopy(self.current_episode_samples[idx][1]),  # t_sys_act
                               copy.deepcopy(self.current_episode_samples[idx][2]),  # t_nlu_v
                               copy.deepcopy(self.current_episode_samples[idx][3]),  # last act
                               copy.deepcopy(self.current_episode_samples[idx][4]),  # sys_act
                               copy.deepcopy(self.current_episode_samples[idx][5]),  # bs vector
                               copy.deepcopy(self.current_episode_samples[idx][6]),  # sys_slot
                               copy.deepcopy(self.current_episode_samples[idx][12]),
                               copy.deepcopy(self.current_episode_samples[idx][7]),  # current act
                               copy.deepcopy(self.current_episode_samples[idx][8]),  # current slot
                               copy.deepcopy(len(self.goal_value)),  # goal len
                               copy.deepcopy(self.current_episode_rewards[idx]),  # reward
                               copy.deepcopy(self.current_episode_samples[idx][4]),  # next manager act
                               copy.deepcopy(self.current_episode_samples[idx][5]),  # next bs
                               copy.deepcopy(self.current_episode_samples[idx][6]),  # next nlu v
                               copy.deepcopy(self.current_episode_samples[idx][12]),  # next encode
                               copy.deepcopy(self.current_episode_samples[idx][9]),  # old policy
                               copy.deepcopy(self.current_episode_samples[idx][10]),  # old slot policy
                    copy.deepcopy(long_return[idx]),
                    1.0))
        self.replay_buffer.append(one_episode_samples)
        self.clear_samples()

    def train_actor_critic(self, entropy_scale=0.001, gamma=1, device=torch.device('cpu')):
        self.nlu.train()
        self.policy.train()
        samples = []
        for one_episode_samples in self.replay_buffer:
            samples.extend(one_episode_samples)
        # nlu
        sys_uttr_batch, pad_mask = self.pad_seq([item.sys_uttr.tolist()[0] for item in samples])
        sys_uttr_batch = torch.tensor(sys_uttr_batch).transpose(0, 1).to(device)
        pad_mask = torch.BoolTensor(pad_mask).to(device)

        t_nlu_v_batch, nlu_mask = self.pad_vector([item.t_nlu_v for item in samples], 3, True)
        t_sys_act_batch = torch.LongTensor([[item.t_sys_act] for item in samples])
        t_nlu_v_batch = torch.LongTensor(t_nlu_v_batch).to(device)
        mask_mat = torch.BoolTensor(nlu_mask).to(device)

        _, p_nlu_vector, logp_sys_act = self.nlu(sys_uttr_batch, pad_mask)
        t_sys_act = torch.zeros(len(samples), len(cfg.sys_act_id2name)).scatter_(1, t_sys_act_batch, 1).to(device)
        nlu_act_loss = -torch.sum(logp_sys_act * t_sys_act)
        maskedNLL = torch.masked_select(torch.gather(torch.log(p_nlu_vector + 1e-21), 2, t_nlu_v_batch.unsqueeze(2)).squeeze(2), mask_mat)
        nlu_slot_loss = -torch.sum(maskedNLL)
        nlu_loss = nlu_act_loss + nlu_slot_loss

        # policy
        # current samples
        last_act_batch = torch.LongTensor([[item.last_act] for item in samples]).to(device)
        sys_act_batch = torch.LongTensor([[item.sys_act] for item in samples]).to(device)
        bs_batch = torch.tensor(self.pad_vector([item.bs_vector.tolist()[0] for item in samples], 3)).to(device)
        nlu_v_batch = torch.tensor(self.pad_vector([item.nlu_v.tolist()[0] for item in samples], 3)).to(device)

        user_act_batch = torch.tensor([[item.user_act] for item in samples]).to(device)
        # user_slot_batch = torch.tensor(self.pad_vector([item.user_slot for item in samples], 0)).to(device)

        if self.net_type == 'RNN' or self.net_type == 'GRU':
            last_encode_batch = torch.tensor([[item.state_encode[0][0] for item in samples]], device=device)
        else:
            last_encode_batch = (torch.tensor([[item.state_encode[0][0][0] for item in samples]], device=device),
                                 torch.tensor([[item.state_encode[1][0][0] for item in samples]], device=device))

        # next_samples
        next_samples = [State(t.user_act, t.next_sys_act, t.next_bs_vector, t.next_nlu_v, t.next_state_encode) for t in samples]
        next_last_act_batch = torch.LongTensor([[item.last_act] for item in next_samples]).to(device)
        next_sys_act_batch = torch.LongTensor([[item.sys_act] for item in next_samples]).to(device)
        next_bs_batch = torch.tensor(self.pad_vector([item.bs_vector.tolist()[0] for item in next_samples], 3)).to(device)
        next_nlu_v_batch = torch.tensor(self.pad_vector([item.nlu_v.tolist()[0] for item in next_samples], 3)).to(device)
        if self.net_type == 'RNN' or self.net_type == 'GRU':
            next_last_encode_batch = torch.tensor([[item.state_encode[0][0] for item in next_samples]], device=device)
        else:
            next_last_encode_batch = (torch.tensor([[item.state_encode[0][0][0] for item in next_samples]], device=device),
                                      torch.tensor([[item.state_encode[1][0][0] for item in next_samples]], device=device))

        # compute current samples
        p_act, p_slot, value, _, _ = self.policy(last_act_batch, sys_act_batch, bs_batch, nlu_v_batch, last_encode_batch)
        log_act = torch.log(p_act + 1e-21)
        log_slot = torch.log(p_slot + 1e-21)

        # compute next samples
        _, _, next_value, _, _ = self.policy(next_last_act_batch, next_sys_act_batch, next_bs_batch, next_nlu_v_batch, next_last_encode_batch)
        reward_batch = torch.Tensor([[item.reward] for item in samples]).to(device)
        terminal_batch = torch.Tensor([[item.terminal] for item in samples]).to(device)
        value_target = reward_batch + (1 - terminal_batch) * next_value * gamma
        # compute loss
        value_loss = F.mse_loss(value, value_target)
        advantage = (value_target - value).detach()
        policy_loss = -torch.sum(advantage * log_act.gather(1, user_act_batch))
        # slot_loss = -torch.sum(torch.gather(torch.log(slot), 2, slot_target.unsqueeze(2)).squeeze(2)) / batch_size
        # slot_loss = -torch.sum(advantage * log_slot.gather(2, user_slot_batch.unsqueeze(2)).squeeze(2))
        slot_batch = torch.LongTensor([item.user_slot for item in samples]).unsqueeze(2).to(device)
        mask_mat = torch.BoolTensor([([True] * item.goal_len + [False] * (cfg.max_chunk - item.goal_len)) for item in samples]).unsqueeze(2).to(device)
        slot_loss = -torch.sum(torch.masked_select(advantage.expand(-1, 3).unsqueeze(2) * log_slot.gather(2, slot_batch), mask_mat))
        entropy_loss = -torch.sum(p_act * log_act)
        # total_loss = value_loss + (nlu_loss + policy_loss + slot_loss - entropy_scale * entropy_loss) / len(samples)
        # total_loss = value_loss + (nlu_loss + policy_loss + slot_loss - entropy_scale * entropy_loss) / len(samples)
        total_loss = value_loss + (policy_loss + slot_loss - entropy_scale * entropy_loss) / len(samples)
        self.nlu_optim.zero_grad()
        self.policy_optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nlu.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.nlu_optim.step()
        self.policy_optim.step()
        # self.nlu.to(torch.device('cpu'))
        # self.policy.to(torch.device('cpu'))
        return total_loss.item()

    def pretrain(self, dataloader, epoch, train_nlu=True, train_policy=True, device=cfg.device):
        self.policy.to(device)
        self.nlu.to(device)
        self.nlu.train()
        self.policy.train()

        pbar = tqdm(dataloader)
        nlu_loss = 0
        policy_loss = 0
        total_nlu_loss = 0
        total_policy_loss = 0

        for idx, data_batch in enumerate(pbar):
            batch_size = len(data_batch[0])
            uttr, pad_mask, max_len, sys_act, nlu_vector, nlu_mask, last_act, bs_vector, user_act, policy_vector, policy_mask = data_batch
            if self.net_type == 'LSTM':
                state_hidden = (torch.zeros(1, batch_size, cfg.user_state_hidden_dim).to(device),
                                torch.zeros(1, batch_size, cfg.user_state_hidden_dim).to(device))
            else:
                state_hidden = torch.zeros(1, batch_size, cfg.user_state_hidden_dim).to(device)

            if train_nlu:
                # train nlu
                uttr = uttr.transpose(0, 1).to(device)
                pad_mask = pad_mask.to(device)
                _, p_nlu_vector, logp_sys_act = self.nlu(uttr, pad_mask)

                t_sys_act = torch.zeros(batch_size, len(cfg.sys_act_id2name)).scatter_(1, sys_act.unsqueeze(1), 1).to(device)
                nlu_act_loss = -torch.sum(logp_sys_act * t_sys_act) / batch_size

                t_nlu_vector = nlu_vector.to(device)
                mask_mat = nlu_mask.to(device)
                maskedNLL = torch.masked_select(torch.gather(torch.log(p_nlu_vector + 1e-21), 2, t_nlu_vector.unsqueeze(2)).squeeze(2), mask_mat)
                nlu_slot_loss = -torch.sum(maskedNLL) / batch_size

                nlu_loss = nlu_act_loss + nlu_slot_loss
                self.nlu_optim.zero_grad()
                nlu_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nlu.parameters(), max_norm=1.0)
                self.nlu_optim.step()
                total_nlu_loss += nlu_loss.item()

            if train_policy:
                # train policy
                # sys_act = torch.zeros(batch_size, len(cfg.sys_act_id2name), dtype=torch.int64).scatter_(1, sys_act.unsqueeze(1),1).to(device)
                sys_act = torch.LongTensor(sys_act).unsqueeze(1).to(device)
                last_act = torch.LongTensor(last_act).unsqueeze(1).to(device)
                nlu_vector = nlu_vector.to(device)
                bs_vector = bs_vector.to(device)
                p_user_act, p_user_slot, _, _, state_hidden = self.policy(last_act, sys_act, bs_vector, nlu_vector, state_hidden)

                if self.net_type == 'LSTM':
                    state_hidden = (state_hidden[0].detach(), state_hidden[1].detach())
                else:
                    state_hidden = state_hidden.detach()

                t_user_act = torch.zeros(batch_size, len(cfg.user_act_id2name)).scatter_(1, user_act.unsqueeze(1), 1).to(device)
                policy_act_loss = -torch.sum(torch.log(p_user_act + 1e-21) * t_user_act) / batch_size
                t_policy_vector = policy_vector.to(device)
                mask_mat = policy_mask.to(device)
                maskedNLL = torch.masked_select(torch.gather(torch.log(p_user_slot + 1e-21), 2, t_policy_vector.unsqueeze(2)).squeeze(2), mask_mat)
                policy_slot_loss = -torch.sum(maskedNLL) / batch_size

                policy_loss = policy_slot_loss + policy_act_loss
                self.policy_optim.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.policy_optim.step()
                total_policy_loss += policy_loss.item()

            pbar.set_description(f'Epoch: {epoch + 1}, Idx: {idx + 1}')
            pbar.set_postfix(nlu_loss=round(float(nlu_loss), 2), p_loss=round(float(policy_loss), 2))
            # pbar.set_postfix(nlu_loss=round(nlu_loss.item(), 2), a_loss=round(policy_act_loss.item(), 2), s_loss=round(policy_slot_loss.item(), 2))
            # pbar.set_postfix(nlu_loss=round(nlu_loss.item(), 2))
        return total_nlu_loss, total_policy_loss

    def evaluate(self, dataloader, test_nlu=True, test_policy=True, device=cfg.device):
        all_count = 0
        nlu_act_r = 0
        nlu_v_r = 0
        policy_act_r = 0
        policy_act_f1 = 0
        policy_v_r = 0

        self.nlu.eval()
        self.policy.eval()
        act_predicted = []
        act_target = []

        for data_batch in dataloader:
            batch_size = len(data_batch[0])
            all_count += batch_size
            uttr, pad_mask, max_len, sys_act, nlu_vector, nlu_mask, last_act, bs_vector, user_act, policy_vector, policy_mask = data_batch

            if test_nlu:
                # test nlu
                uttr = uttr.transpose(0, 1).to(device)
                pad_mask = pad_mask.to(device)
                p_sys_act, p_nlu_vector, _ = self.nlu(uttr, pad_mask)
                p_sys_act = torch.argmax(p_sys_act, dim=-1).to('cpu')
                p_nlu_vector = torch.argmax(p_nlu_vector, dim=-1).to('cpu')

                nlu_act_r += torch.sum(p_sys_act == sys_act).item()
                nlu_v_r += sum(torch.equal(i, j) for i, j in zip(p_nlu_vector, nlu_vector))

            if test_policy:
                # t_sys_act = torch.LongTensor(sys_act).unsqueeze(1).to(device)
                sys_act = torch.LongTensor(sys_act).unsqueeze(1).to(device)
                last_act = torch.LongTensor(last_act).unsqueeze(1).to(device)
                nlu_vector = nlu_vector.to(device)
                bs_vector = bs_vector.to(device)
                if self.net_type == 'LSTM':
                    state_hidden = (torch.zeros(1, batch_size, cfg.user_state_hidden_dim).to(device),
                                    torch.zeros(1, batch_size, cfg.user_state_hidden_dim).to(device))
                else:
                    state_hidden = torch.zeros(1, batch_size, cfg.user_state_hidden_dim).to(device)
                p_user_act, p_user_slot, _, _, _ = self.policy(last_act, sys_act, bs_vector, nlu_vector, state_hidden)
                p_user_act = torch.argmax(p_user_act, dim=-1).to('cpu')
                p_user_slot = torch.argmax(p_user_slot, dim=-1).to('cpu')

                policy_act_r += torch.sum(p_user_act == user_act).item()
                act_predicted += p_user_act
                act_target += user_act
                policy_v_r += sum(torch.equal(i, j) for i, j in zip(p_user_slot, policy_vector))

        nlu_act_acc, nlu_v_acc = nlu_act_r / all_count, nlu_v_r / all_count
        policy_act_acc, policy_v_acc = policy_act_r / all_count, policy_v_r / all_count
        policy_act_f1 += f1_score(act_target, act_predicted, average='macro')

        logging.info(
            'nlu_act_acc:{:.4f} nlu_slot_acc:{:.4f} policy_act_acc:{:.4f} policy_act_f1:{:.4f} policy_slot_acc:{:.4f}'.format(nlu_act_acc, nlu_v_acc, policy_act_acc, policy_act_f1, policy_v_acc))
        return nlu_act_acc, nlu_v_acc, policy_act_f1, policy_v_acc

    def train_policy_iter(self, data_batch, device=cfg.device):
        self.policy.to(device)
        self.policy.train()
        batch_loss = 0
        batch_size = data_batch[0]['last_act'].size(0)
        if self.net_type == 'LSTM':
            encode_tensor = (torch.zeros(1, batch_size, cfg.user_state_hidden_dim).to(device),
                             torch.zeros(1, batch_size, cfg.user_state_hidden_dim).to(device))
        else:
            encode_tensor = torch.zeros(1, batch_size, cfg.user_state_hidden_dim).to(device)

        for turn_idx, batch in enumerate(data_batch):
            last_act, sys_act, sys_slot, bs_vector, policy_act, policy_slot, policy_mask = \
                batch['last_act'], batch['sys_act'], batch['sys_slot'], batch['bs_vector'], \
                batch['policy_act'], batch['policy_slot'], batch['policy_mask']
            last_act = last_act.to(device)
            sys_act = sys_act.to(device)
            sys_slot = sys_slot.to(device)
            bs_vector = bs_vector.to(device)

            p_user_act, p_user_slot, _, _, encode_tensor = self.policy(last_act, sys_act, bs_vector, sys_slot, encode_tensor)
            encode_tensor = encode_tensor.detach()
            valid_turn = batch['valid'].to(device)

            policy_target = torch.zeros(batch_size, len(cfg.user_act_id2name)).scatter_(1, policy_act, 1).to(device)
            policy_loss = -torch.sum(torch.sum(torch.log(p_user_act + 1e-21) * policy_target, dim=1) * valid_turn) / batch_size

            policy_slot = policy_slot.to(device)
            mask_mat = policy_mask.to(device)
            maskedNLL = torch.masked_select(torch.gather(torch.log(p_user_slot + 1e-21), 2, policy_slot.unsqueeze(2)).squeeze(2), mask_mat)
            policy_slot_loss = -torch.sum(maskedNLL) / batch_size

            loss = policy_loss + policy_slot_loss

            self.policy_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10)
            self.policy_optim.step()
            batch_loss += loss.item()
        return batch_loss / len(data_batch)

    def evaluate_policy_iter(self, data_batch, device=cfg.device):
        self.policy.to(device)
        self.policy.eval()
        policy_act_correct = 0
        policy_slot_correct = 0
        total_num = 0

        batch_size = data_batch[0]['last_act'].size(0)
        if self.net_type == 'LSTM':
            encode_tensor = (torch.zeros(1, batch_size, cfg.user_state_hidden_dim).to(device),
                             torch.zeros(1, batch_size, cfg.user_state_hidden_dim).to(device))
        else:
            encode_tensor = torch.zeros(1, batch_size, cfg.user_state_hidden_dim).to(device)

        for turn_idx, batch in enumerate(data_batch):
            last_act, sys_act, sys_slot, bs_vector, policy_act, policy_slot, policy_mask = \
                batch['last_act'], batch['sys_act'], batch['sys_slot'], batch['bs_vector'], \
                batch['policy_act'], batch['policy_slot'], batch['policy_mask']
            last_act = last_act.to(device)
            sys_act = sys_act.to(device)
            sys_slot = sys_slot.to(device)
            bs_vector = bs_vector.to(device)

            p_user_act, p_user_slot, _, _, encode_tensor = self.policy(last_act, sys_act, bs_vector, sys_slot, encode_tensor)

            valid_turn = batch['valid'].to(device)
            policy_act = policy_act.to(device)
            policy_slot = policy_slot.to(device)

            policy = torch.argmax(p_user_act, dim=-1)
            policy_act_correct += ((policy == policy_act.squeeze(1)) * valid_turn).sum().item()

            slot = torch.argmax(p_user_slot, dim=-1)
            policy_slot_correct += sum(torch.equal(i, j) for i, j, k in zip(slot, policy_slot, valid_turn) if k)

            total_num += valid_turn.sum().item()

        return policy_act_correct / total_num, policy_slot_correct / total_num

    def evaluate_policy(self, dataloader, epoch, device=cfg.device):
        self.policy.eval()
        pbar = tqdm(dataloader)
        policy_act_acc, policy_slot_acc = 0, 0
        for i, data_batch in enumerate(pbar):
            policy_iter_f1, slot_iter_acc = self.evaluate_policy_iter(data_batch, device)
            policy_act_acc += policy_iter_f1
            policy_slot_acc += slot_iter_acc

            pbar.set_description(f'Dev Epoch: {epoch}, Idx: {i + 1}')
        return policy_act_acc / len(dataloader), policy_slot_acc / len(dataloader)

    def save_model(self, save_path=None, save_nlu=True, save_policy=True):
        if save_path is None:
            save_path = os.path.join(cfg.model_save_path, 'simulator')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info(f'user model saved at {save_path}, save nlu {save_nlu}, save policy {save_policy}.')

        if save_nlu:
            torch.save(self.nlu.state_dict(), save_path + '/nlu.pkl')
        if save_policy:
            torch.save(self.policy.state_dict(), save_path + '/policy.pkl')

    def load_model(self, load_path=None, load_nlu=True, load_policy=True):
        if load_path is None:
            load_path = os.path.join(cfg.model_save_path, 'simulator')

        logging.info(f'user model load from {load_path}, load nlu {load_nlu}, load policy {load_policy}.')
        if load_nlu:
            self.nlu.load_state_dict(torch.load(load_path + '/nlu.pkl', map_location=cfg.device))
        if load_policy:
            self.policy.load_state_dict(torch.load(load_path + '/policy.pkl', map_location=cfg.device))


if __name__ == '__main__':
    pass
