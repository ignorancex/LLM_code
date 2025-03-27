import logging
import os
import torch
import random
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--description', type=str, default="test")
parser.add_argument('--device', type=str)
parser.add_argument('--kl_scale', type=float)
parser.add_argument('--seed', type=int)
parser.add_argument('--save_log', type=bool)
parser.add_argument('--use_tfboard', type=bool)
parser.add_argument('--simulator_num', type=int)
parser.add_argument('--rule_sim', type=bool)


class Config(object):
    def __init__(self):
        self.file_folder = 'data/SSD_phone'
        self.train_path = os.path.join(self.file_folder, 'train.json')
        self.dev_path = os.path.join(self.file_folder, 'dev.json')
        self.test_path = os.path.join(self.file_folder, 'test.json')
        self.vocab_path = 'src/corpus/vocab.json'
        self.tgt_vocab_path = 'src/corpus/vocab.json'
        self.multi_user_template_path = 'src/corpus/multiuser.json'

        # experiment settings
        self.seed = 5
        self.MAX_STATE_LEN = 20
        self.MAX_STATE_BLOCK_NUM = 5
        self.MAX_UTTER_LEN = 30
        self.algo = 'a2c'
        # rei, rwb, a2c, a3c, ppo

        self.num_processor = 4
        self.share_user = True

        self.max_turn = 24
        self.max_chunk = 3
        self.patience = 6
        self.device = 'cuda:0'
        # self.device = 'cpu'
        self.batch_size = 128
        self.RegularExpressions = ['^1\d{10}$', '^1\d{1,}']
        self.model_save_path = 'caches/simulator/314'
        self.save_log = True
        self.simulator_num = 1
        self.rule_sim = False
        self.manager_pretrain = True
        self.sim_pretrain = True
        self.pretrain_value_net = False
        self.manager_pre_path = 'caches/manager/sl-413_add_value_net/manager_model.pkl'
        self.sim_pre_path = ['caches/simulator/415', 'caches/simulator/415', 'caches/simulator/415']
        self.rl_train_step = 100000
        self.test_frequence = 100
        self.save_frequence = 2000
        self.trainmethod = 'method1'
        self.description = 'test'
        self.use_tfboard = True

        # manager training settings
        self.manager_emb_dim = 100
        self.manager_char_dim = 64
        self.manager_hidden_dim = 64
        self.manager_state_dim = 64
        self.manager_hidden_one_dim = 32
        self.manager_hidden_two_dim = 32
        self.manager_sl_learning_rate = 1e-3
        self.manager_rl_learning_rate = 1e-5
        self.manager_entropy_scale = 0.01
        self.manager_nhead = 8
        self.manager_dhid = self.manager_emb_dim * 4
        self.manager_encoder_layers = 6
        self.manager_decoder_layers = 6
        self.manager_dropout = 0.1

        self.manager_replay_buffer_size = 6

        # simulator setting
        self.user_error_prob = 0.3
        self.user_replay_episodes_size = 6
        self.user_algo = 'a2c'
        self.simulator_entropy_scale = 0.01
        self.kl_scale = 0.0

        # simulator model settings
        # self.max_uttr_len = 64
        self.max_uttr_len = 128
        self.user_max_len = 512
        self.user_embedding_dim = 128
        self.user_nhead = 8
        self.user_dhid = self.user_embedding_dim * 4
        self.user_nlayers = 4
        self.user_dropout = 0.2
        self.user_policy_embedding_dim = 32
        self.user_policy_hidden_dim = 64
        self.user_state_hidden_dim = 32
        self.user_net_type = 'GRU'
        self.user_sl_lr = 1e-4
        self.user_rl_lr = 1e-5

        self._init_user_act()
        self.user_act_name2id = {act: idx for idx, act in enumerate(self.user_act_id2name)}
        self._init_sys_act()
        self.sys_act_name2id = {act: idx for idx, act in enumerate(self.sys_act_id2name)}
        self._init_special_token()

        # self._init_logging_handler()
        # self._set_seed(self.seed)

    def __str__(self):
        s = ''
        for k, v in self.__dict__.items():
            s += '{} : {}\n'.format(k, v)
        return s

    # @staticmethod
    def set_seed(self):
        seed = self.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _init_special_token(self):
        self.mask_token = '#'
        self.split_token = '$'

    def _init_user_act(self):
        self.user_act_id2name = [
            "offer",
            "inform_normal",
            "inform_multi",
            "inform_2x",
            "inform_tone",
            "inform_update",
            "affirm",
            "deny",
            "update_normal",
            "update_one",
            # "update_before_miss",
            # "update_after_miss",
            # "update_last_miss",
            # "update_negate",
            # "update_negate_n",
            # "update_two_part",
            "update_sure",
            "update_special",
            "update_sub",
            "ack",
            "finish",
            "ask_state",
            "bye",
            "restart",
            "ask_repeat",
            "doubt_identity",
            "how_signal",
            "bad_signal",
            "good_signal",
            "wait",
            "other"
        ]

    def _init_sys_act(self):
        self.sys_act_id2name = ['request', 'continue', 'req_more', 'implicit_confirm', 'explicit_confirm', 'ack',
                                'req_correct',
                                'compare', 'ask_restart', 'bye', 'good_signal', 'how_signal', 'bad_signal', 'repeat',
                                'robot', 'other']

    def init_logging_handler(self, mode='rl'):
        now_time = time.strftime("%m%d_%H%M", time.localtime())
        log_format = "%(asctime)s - [%(filename)s:%(lineno)d]: %(message)s"
        data_format = "%m/%d/%Y %H:%M:%S %p"
        if mode == 'rl':
            filename = './logdir/log_{}_{}_1v{}_rule{}_sd{}_{}.txt'.format(now_time, self.description, self.simulator_num, int(self.rule_sim), self.seed, self.algo)
        elif mode == 'sl_user':
            filename = './logdir/log_sl_user_{}_{}_sd{}.txt'.format(now_time, self.description, self.seed)
        elif mode == 'rl_data':
            filename = './logdir/log_rl_data_{}_{}_sd{}.txt'.format(now_time, self.description, self.seed)
        elif mode == 'sl_sys':
            filename = './logdir/log_sl_sys_{}_{}_sd{}.txt'.format(now_time, self.description, self.seed)
        else:
            filename = './logdir/log_{}_{}_{}_sd{}.txt'.format(mode, now_time, self.description, self.seed)

        stderr_handler = logging.StreamHandler()

        if not os.path.exists('./logdir'):
            os.mkdir('./logdir')
        if self.save_log:
            file_handler = logging.FileHandler(filename)
            logging.basicConfig(handlers=[stderr_handler, file_handler], format=log_format, datefmt=data_format)
        else:
            logging.basicConfig(handlers=[stderr_handler], format=log_format, datefmt=data_format)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.info(self.__str__())


global_config = Config()
