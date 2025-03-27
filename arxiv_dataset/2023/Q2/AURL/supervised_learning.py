import logging
import os
import shutil
import time
import torch
from tqdm import tqdm
from src.managers.Ross_manager import RossManager
from src.system.interlocution import Interlocution
from src.dataLoader.new_dataloder import read_dialogue_from_data, data_loader
from test_and_analyse import evaluate_dialog_success
from src.utilities.config_new import global_config as cfg
import logging
from torch.utils.tensorboard import SummaryWriter

if cfg.device != 'cpu' and torch.cuda.is_available():
    device = torch.device(cfg.device)
else:
    device = torch.device('cpu')
early_stop = None


def train_manager():
    # initialize managers
    manager = RossManager(emb_dim=cfg.manager_emb_dim, char_dim=cfg.manager_char_dim, state_dim=cfg.manager_state_dim,
                          hidden_size=cfg.manager_hidden_dim, hidden_one_dim=cfg.manager_hidden_one_dim, hidden_two_dim=cfg.manager_hidden_two_dim,
                          sim_action_num=len(cfg.user_act_id2name), algo=cfg.algo, lr=cfg.manager_sl_learning_rate
                          )
    manager.trainable = True
    manager.net.train()
    checkpoints_path = './caches/manager/sl-{}/'.format(cfg.description)

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

    if cfg.use_tfboard:
        logging.info('using tensorboard to recode results.')
        writer = SummaryWriter(checkpoints_path)


    train_manager_dataloader = data_loader('./data/SSD_phone/train.json', './src/corpus/vocab.json', cfg.batch_size)
    dev_manager_dataloader = data_loader('./data/SSD_phone/dev.json', './src/corpus/vocab.json', cfg.batch_size)
    test_dialogs = read_dialogue_from_data('./data/SSD_phone/test.json')

    inter = Interlocution()
    avg_best, cnt = 0, 0
    start_time = time.time()

    for epoch in range(200):
        # train
        pbar = tqdm(train_manager_dataloader)
        manager.net.to(device)
        manager.net.train()
        total_loss = 0

        for i, data_batch in enumerate(pbar):
            loss = manager.train_iter(data_batch, ['号-号'], device=device)
            pbar.set_description(f'Epoch: {epoch}, Idx: {i + 1}')
            pbar.set_postfix(loss=round(loss, 2))
            total_loss += loss

        # dev
        end_time = time.time()
        logging.info('train step:{}\ttime:{}'.format((epoch+1), end_time - start_time))
        start_time = time.time()
        manager.net.eval()
        intent_acc, state_acc, policy_f1, slot_acc = manager.evaluate(dev_manager_dataloader, epoch, ['号-号'], device=device)
        manager.scheduler.step(state_acc)
        logging.info('\nepoch:{}\tintent F1: {:.3f}\tstate ACC: {:.3f}\tpolicy_f1: {:.3f}\tslot acc: {:.3f}'
                    .format(epoch, intent_acc, state_acc, policy_f1, slot_acc))
        # results = (intent_acc, state_acc, policy_f1)
        # with open(checkpoints_path + 'test_{}.json'.format(epoch), 'w') as f:
        #     json.dump(results, f)
        # results = state_acc * policy_f1
        results = state_acc

        if cfg.use_tfboard:
            writer.add_scalar("Train/Loss", total_loss, epoch + 1)
            writer.add_scalar("Test/intent_acc", intent_acc, epoch + 1)
            writer.add_scalar("Test/state_acc", state_acc, epoch + 1)
            writer.add_scalar("Test/policy_f1", policy_f1, epoch + 1)
            writer.add_scalar("Test/slot_acc", slot_acc, epoch + 1)

        if results >= avg_best:
            cnt = 0
            avg_best = results
            best_model = manager.net
        else:
            cnt += 1

        if cnt == cfg.patience or (results == 1.0 and early_stop is None):
            logging.info("Ran out of patient, early stop...")
            break

    # find dev best
    torch.save(best_model.state_dict(), checkpoints_path + 'manager_model.pkl')
    # manager.net.load_state_dict(torch.load('caches/manager/sl-407_sys_add_last_act/manager_model.pkl', map_location=torch.device('cpu')))
    #
    # if cfg.use_tfboard:
    #     writer.close()

    # test
    manager.net = best_model
    manager.net.eval()
    manager.net.to('cpu')
    dialog_success, block_acc, state_acc, policy_f1, slot_acc = evaluate_dialog_success(test_dialogs, manager, inter)
    print('Test result: dst slot acc: {:.4f}\tdst joint acc: {:.4f}\tdialog success acc: {:.4f}\tslot acc: {:.4f}\tpolicy f1: {:.4f}'
          .format(block_acc, state_acc, dialog_success, slot_acc, policy_f1))
    logging.info('Test result: dst slot acc: {:.4f}\tdst joint acc: {:.4f}\tdialog success acc: {:.4f}\tslot acc: {:.4f}\tpolicy f1: {:.4f}'
          .format(block_acc, state_acc, dialog_success, slot_acc, policy_f1))


if __name__ == '__main__':
    manager = RossManager(emb_dim=cfg.manager_emb_dim, char_dim=cfg.manager_char_dim, state_dim=cfg.manager_state_dim,
                          hidden_size=cfg.manager_hidden_dim, hidden_one_dim=cfg.manager_hidden_one_dim, hidden_two_dim=cfg.manager_hidden_two_dim,
                          sim_action_num=len(cfg.user_act_id2name), algo=cfg.algo, lr=cfg.manager_sl_learning_rate
                          )
    manager.trainable = True
    manager.net.train()
    test_dialogs = read_dialogue_from_data('./data/SSD_phone/test.json')

    inter = Interlocution()
    manager.net.to('cpu')
    manager.net.load_state_dict(torch.load('./caches/manager/sl-520/manager_model.pkl', map_location=torch.device('cpu')))
    # manager.net.load_state_dict(torch.load(cfg.manager_pre_path, map_location=torch.device('cpu')))
    # test_manager_dataloader = data_loader('./data/SSD_phone/test.json', './src/corpus/vocab.json', cfg.batch_size)

    # intent_acc, state_acc, policy_f1, slot_acc = manager.evaluate(test_manager_dataloader, 0, ['号-号'], device=cfg.device)
    # print('\nepoch:{}\tintent F1: {:.3f}\tstate ACC: {:.3f}\tpolicy_f1: {:.3f}\tslot acc: {:.3f}'
    #              .format(0, intent_acc, state_acc, policy_f1, slot_acc))
    dialog_success, block_acc, state_acc, policy_f1, slot_acc = evaluate_dialog_success(test_dialogs, manager, inter)
    # inter.save_log('error.log')
    print('Test result: dst slot acc: {:.4f}\tdst joint acc: {:.4f}\tdialog success acc: {:.4f}\tslot acc: {:.4f}\tpolicy f1: {:.4f}'
          .format(block_acc, state_acc, dialog_success, slot_acc, policy_f1))
