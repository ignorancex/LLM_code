import copy
import json
import logging
import os
import random
import shutil
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from src.managers.Ross_manager import RossManager
from src.simulators.multiusersimulator import MultiSimulator
from src.simulators.weighted_rule_simulator import WeightedRuleSimulator
from src.system.interlocution import Interlocution
from src.utilities.config_new import global_config as cfg


def train():
    if cfg.device != 'cpu' and torch.cuda.is_available():
        device = torch.device(cfg.device)
    else:
        device = torch.device('cpu')

    # initialize simulators:
    sim_group = []
    sim_model_num = cfg.simulator_num
    for it in range(sim_model_num):
        sim = MultiSimulator()
        sim_group.append(sim)

    if cfg.sim_pretrain:
        for i, sim_path in enumerate(cfg.sim_pre_path):
            if i == len(sim_group):
                break
            if os.path.exists(sim_path):
                if sim_group[i].trainable:
                    sim_group[i].load_model(sim_path)
                    logging.info('simulator {} loaded model from {}'.format(i, sim_path))
                else:
                    break
            else:
                logging.info('simulator path not exists')
                exit()
    # initialize manager:
    manager = RossManager(emb_dim=cfg.manager_emb_dim, char_dim=cfg.manager_char_dim, state_dim=cfg.manager_state_dim,
                          hidden_size=cfg.manager_hidden_dim, hidden_one_dim=cfg.manager_hidden_one_dim, hidden_two_dim=cfg.manager_hidden_two_dim,
                          sim_action_num=len(cfg.user_act_id2name), algo=cfg.algo, lr=cfg.manager_rl_learning_rate)
    if cfg.manager_pretrain:
        if os.path.exists(cfg.manager_pre_path):
            manager.net.load_state_dict(torch.load(cfg.manager_pre_path, map_location=torch.device(cfg.device)))
            logging.info('system loaded model from {}'.format(cfg.manager_pre_path))
        else:
            logging.info('manager path not exist')
            exit()

    checkpoints_path = './caches/cooperate/{}-{}-{}/seed-{}/'.format(cfg.simulator_num, cfg.algo, cfg.description, cfg.seed)
    logging.info(f'checkpoints_path is {checkpoints_path}')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    else:
        try:
            shutil.rmtree(checkpoints_path)
        except Exception as e:
            logging.error('delete checkpoints failed, reason is {}.'.format(e))
        try:
            os.makedirs(checkpoints_path)
        except:
            pass

    inter = Interlocution()
    results = dict()
    sys_loss = 0
    user_loss = 0
    user = 0
    best_acc = 0.0
    best_epoch = 0
    first_results = []
    for i, sim in enumerate(sim_group):
        if sim.trainable == False:
            result = inter.test_episodes(manager, sim, 100, 'rule')
        else:
            result = inter.test_episodes(manager, sim, 100, 'model')
        first_results.append(result)
    if cfg.use_tfboard:
        logging.info('using tensorboard to recode results.')
        writer = SummaryWriter(checkpoints_path)
        for i, r in enumerate(first_results):
            writer.add_scalar(f"Test/Succ_{i + 1}", r[1], 0)
            writer.add_scalar(f"Test/sys_rwd_{i + 1}", r[3][0], 0)
            writer.add_scalar(f"Test/usr_rwd_{i + 1}", r[3][1], 0)

    logging.info('frequence:0\ttest results:{}'.format(first_results))

    for i, sim in enumerate(sim_group):
        sim_ver = 'model' if sim.trainable else 'rule'
        inter.play_one_episode_for_test(manager, sim, 0, sim_ver, save_log_path=checkpoints_path + 'log_0_{}.json'.format(i))

    if cfg.pretrain_value_net:
        dialogs = json.load(open('./data/SSD_phone/train.json', 'r', encoding='utf-8'))
        all_keys = list(dialogs.keys())
        for step in range(4000):
            for it in range(6):
                inter.add_one_episode_for_sys_train(manager, dialogs[random.choice(all_keys)])

            if manager.trainable:
                value_loss = manager.update_value_net(entropy_scale=cfg.manager_entropy_scale, device=device)
            if cfg.use_tfboard:
                writer.add_scalar(f"Train/value_loss", value_loss, step+1)

    # try:
    start_time = time.time()
    for step in range(cfg.rl_train_step):
        manager.net.train()
        if cfg.trainmethod == 'method1':
            # train1
            for it in range(int(6 / cfg.simulator_num)):
                for sim in sim_group:
                    inter.play_one_episode_for_train(manager, sim)
            if manager.trainable:
                if cfg.algo == 'rwb':
                    sys_loss += manager.train_reinforce_with_baseline(entropy_scale=cfg.manager_entropy_scale, device=device)
                elif cfg.algo == 'a2c':
                    sys_loss += manager.train_actor_critic(entropy_scale=cfg.manager_entropy_scale, device=device)
                elif cfg.algo == 'ppo':
                    sys_loss += manager.train_ppo(entropy_scale=cfg.manager_entropy_scale, device=device)
                else:
                    pass

            for sim in sim_group:
                if sim.trainable:
                    if cfg.algo == 'rwb':
                        user_loss += sim.train_reinforce_with_baseline(entropy_scale=cfg.simulator_entropy_scale, device=device)
                    elif cfg.algo == 'a2c':
                        user_loss += sim.train_actor_critic(entropy_scale=cfg.simulator_entropy_scale, device=device)
                    elif cfg.algo == 'ppo':
                        user_loss += sim.train_ppo(entropy_scale=cfg.simulator_entropy_scale, device=device)
                    else:
                        pass

        if (step+1) % 500 == 0:
            easy_rate, middle_rate, hard_rate = manager.train_dst(device)
            if cfg.use_tfboard:
                writer.add_scalar("Curriculum/easy", easy_rate, step + 1)
                writer.add_scalar("Curriculum/middle", middle_rate, step + 1)
                writer.add_scalar("Curriculum/hard", hard_rate, step + 1)

        # test
        if (step + 1) % cfg.test_frequence == 0:
            manager.net.eval()
            end_time = time.time()
            logging.info('train step:{}\ttime:{}\tbest_opoch:{}'.format((step + 1), end_time - start_time, best_epoch))

            fre_num = (step + 1) // cfg.test_frequence
            for i, sim in enumerate(sim_group):
                sim_ver = 'model' if sim.trainable else 'rule'
                inter.play_one_episode_for_test(manager, sim, 0, sim_ver, save_log_path=checkpoints_path + 'log_{}_{}.json'.format(fre_num, i))

            dev_succ, test_succ = 0.0, 0.0
            dev_result = []
            for sim in sim_group:
                sim_ver = 'model' if sim.trainable else 'rule'
                dev_results = inter.test_episodes(manager, sim, 100, sim_ver)
                dev_succ += dev_results[1]
                dev_result.append(dev_results)

            if dev_succ >= best_acc:
                best_acc = dev_succ
                best_epoch = step + 1
                if manager.trainable:
                    manager.save(checkpoints_path + 'manager_model_best.pkl')
                for i, sim in enumerate(sim_group):
                    if sim.trainable:
                        sim.save_model(checkpoints_path + 'simulator_model_best_{}'.format(i))

            results[fre_num] = copy.deepcopy([dev_result])
            logging.info('frequence:{}\ttest results:{}'.format(fre_num, results[fre_num]))
            with open(checkpoints_path + 'test_{}.json'.format(fre_num), 'w') as f:
                json.dump(results[fre_num], f)

            if cfg.use_tfboard:
                writer.add_scalar("Train/sys_Loss", sys_loss / cfg.test_frequence, step + 1)
                writer.add_scalar("Train/user_Loss", user_loss / cfg.test_frequence, step + 1)

                for i, r in enumerate(dev_result):
                    writer.add_scalar(f"Test/Succ_{i + 1}", r[1], step + 1)
                    writer.add_scalar(f"Test/sys_rwd_{i + 1}", r[3][0], step + 1)
                    writer.add_scalar(f"Test/usr_rwd_{i + 1}", r[3][1], step + 1)
                    writer.add_scalar(f"Test/path_{i + 1}", r[4], step + 1)

            # loss = 0
            sys_loss = 0
            user_loss = 0
            start_time = time.time()
            if (step + 1) % cfg.save_frequence == 0:
                if manager.trainable:
                    manager.save(checkpoints_path + 'manager_model_{}.pkl'.format(fre_num))
                for i, sim in enumerate(sim_group):
                    if sim.trainable:
                        sim.save_model(checkpoints_path + 'simulator_model_{}_{}'.format(fre_num, i))
    # except (KeyboardInterrupt, Exception) as e:
    #     logging.error('process shutdown, reason is {}'.format(e))
    #     if cfg.use_tfboard:
    #         writer.close()
    #     if manager.trainable:
    #         manager.save(checkpoints_path + 'manager_model_final.pkl')
    #     for i, sim in enumerate(sim_group):
    #         if sim.trainable:
    #             sim.save_model(checkpoints_path + 'simulator_model_final_{}'.format(i))
    #     exit(-1)


if __name__ == '__main__':
    train()
