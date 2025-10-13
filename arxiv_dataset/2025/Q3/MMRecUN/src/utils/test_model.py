# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os
import torch

def test(model, dataset, config_dict, unlearn=None, name=None):
    # merge config dict
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    
    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, forget_dataset, valid_dataset, test_dataset = dataset.split('gold_user_item')
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Forget====\n' + str(forget_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))
    forget_data = EvalDataLoader(config, forget_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
    # forget_data_trainer = TrainDataLoader(config, forget_dataset, batch_size=config['train_batch_size'], shuffle=True)
    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        # train_data.pretrain_setup()
        # model loading and initialization
        if isinstance(name, str):
            model = get_model(config['model'])(config, train_data).to(config['device'])
            model.load_state_dict(torch.load(config['checkpoint_dir'] + name))
            model.eval()
        else:
            model = name
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer()(config, model)
        _, valid_result = trainer._valid_epoch(valid_data)
        # test
        _, test_result = trainer._valid_epoch(test_data)
        # forget
        _, forget_result = trainer._valid_epoch(forget_data)

        trainer.logger.info('valid result: \n' + dict2str(valid_result))
        trainer.logger.info('test result: \n' + dict2str(test_result))
        trainer.logger.info('forget result: \n' + dict2str(forget_result))
   