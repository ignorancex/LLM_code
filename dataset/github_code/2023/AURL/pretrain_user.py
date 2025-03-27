import json
import logging
import os.path
import shutil

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utilities.config_new import global_config as cfg
from src.dataLoader.user_dataset import data_loader
# from src.dataLoader.user_dataset_2in1 import data_loader
from src.simulators.multiusersimulator import MultiSimulator
from test_and_analyse import evaluate_simulator, evaluate_simulator2
from src.dataLoader.user_dataset_policy import data_loader_policy
import logging

def pretrain_nlu():
    sim = MultiSimulator('sl')
    train_dataloader = data_loader('train')
    dev_dataloader = data_loader('dev')

    nlu_best = 0.
    nlu_cnt = 0

    t_nlu = True

    checkpoints_path = os.path.join('caches/simulator/', cfg.description)
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    # if cfg.use_tfboard:
    #     logging.info('using tensorboard to recode results.')
    #     writer = SummaryWriter(checkpoints_path)

    for i in range(200):
        nlu_loss, policy_loss = sim.pretrain(train_dataloader, i, train_nlu=t_nlu, train_policy=False)
        nlu_act_acc, nlu_v_acc, policy_act_acc, policy_v_acc = sim.evaluate(dev_dataloader, t_nlu, False)

        logging.info('\nepoch:{}\tnlu act acc: {:.4f}\tnlu slot acc: {:.4f}'.format(i, nlu_act_acc, nlu_v_acc))

        nlu_result = nlu_act_acc * nlu_v_acc
        if nlu_result >= nlu_best:
            nlu_cnt = 0
            nlu_best = nlu_result
            best_nlu_model = sim.nlu
        else:
            nlu_cnt += 1

        if nlu_cnt == cfg.patience or nlu_result == 1.0:
            sim.nlu = best_nlu_model
            sim.save_model(save_path=os.path.join('caches/simulator/', cfg.description), save_policy=False)
            logging.info("Ran out of patient, early stop...")
            break

        # if cfg.use_tfboard:
        #     if t_nlu:
        #         writer.add_scalar("Train/nlu_Loss", nlu_loss, i + 1)
        #         writer.add_scalar("Test/nlu_act_acc", nlu_act_acc, i + 1)
        #         writer.add_scalar("Test/nlu_slot_acc", nlu_v_acc, i + 1)
        #
        # if not t_nlu:
        #     break

def pretrain_policy():
    sim = MultiSimulator('sl')
    train_dataloader = data_loader_policy('train')
    dev_dataloader = data_loader_policy('dev')

    policy_best = 0.
    policy_cnt = 0

    checkpoints_path = os.path.join('caches/simulator/', cfg.description)
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    # if cfg.use_tfboard:
    #     logging.info('using tensorboard to recode results.')
    #     writer = SummaryWriter(checkpoints_path)

    for epoch in range(200):
        pbar = tqdm(train_dataloader)
        total_loss = 0
        for i, data_batch in enumerate(pbar):
            loss = sim.train_policy_iter(data_batch)
            pbar.set_description(f'Epoch: {epoch}, Idx: {i + 1}')
            pbar.set_postfix(loss=round(loss, 2))
            total_loss += loss

        policy_f1, slot_acc = sim.evaluate_policy(dev_dataloader, epoch)
        logging.info('\nepoch:{}\tpolicy_f1: {:.4f}\tslot acc: {:.4f}'.format(epoch, policy_f1, slot_acc))

        policy_result = policy_f1 * slot_acc
        if policy_result >= policy_best:
            policy_cnt = 0
            policy_best = policy_result
            best_policy_model = sim.policy
        else:
            policy_cnt += 1

        if policy_cnt == cfg.patience or policy_result == 1.0:
            sim.policy = best_policy_model
            sim.save_model(save_path=os.path.join('caches/simulator/', cfg.description), save_nlu=False)
            logging.info("Ran out of patient, early stop...")
            break


def pretrain_user(pre_nlu=True, pre_policy=True):
    if pre_nlu:
        pretrain_nlu()
    if pre_policy:
        pretrain_policy()
    sim = MultiSimulator('sl')
    # sim.nlu.to('cpu')
    # sim.policy.to('cpu')
    sim.load_model(os.path.join('caches/simulator/', cfg.description), load_policy=False)
    sim.load_model(os.path.join('caches/simulator/414'), load_nlu=False)
    nlu_a, nlu_v, p_a, p_af1, p_v = evaluate_simulator(json.load(open(cfg.test_path, 'r', encoding='utf-8')), sim)
    logging.info('Test result: nlu act acc: {:.4f}\tnlu slot acc: {:.4f}\tpolicy act acc: {:.4f}\tpolicy act f1: {:.4f}\tpolicy vector acc: {:.4f}'
          .format(nlu_a, nlu_v, p_a, p_af1, p_v))



if __name__ == '__main__':
    # parser.add_argument('--sl_user', action='store_true', default=False)
    # parser.add_argument('--test_user', action='store_true', default=False)
    # args = parser.parse_args()
    # if args.sl_user:
    # mode = 'sl_user'
    # mode = 'sl_sys'
    # model = 'rl'

    # pretrain_user()
    # if args.test_user:
    sim = MultiSimulator('sl')
    # test_dataloader = data_loader('dev')
    # nlu_act_acc, nlu_v_acc, policy_act_acc, policy_v_acc = sim.evaluate(test_dataloader, test_nlu=True, test_policy=False)
    # print('nlu act acc: {:.4f}\tnlu slot acc: {:.4f}'.format( nlu_act_acc, nlu_v_acc))

    # sim.load_model(cfg.sim_pre_path[0])
    sim.load_model('caches/simulator/415')
    nlu_a, nlu_v, p_a, p_af1, p_v = evaluate_simulator(json.load(open(cfg.test_path, 'r', encoding='utf-8')), sim, use_gt=False)
    print('Test result: nlu act acc: {:.4f}\tnlu slot acc: {:.4f}\tpolicy act acc: {:.4f}\tpolicy act f1: {:.4f}\tpolicy vector acc: {:.4f}'
                 .format(nlu_a, nlu_v, p_a, p_af1, p_v))

    # sim.evaluate(test_dataloader, False, True)
