import argparse
from logging import getLogger

from recbole.config import Config
from recbole.utils import init_seed, set_color
from logger import init_logger
from utils import create_dataset, data_preparation, create_samplers, obtain_mask, get_dataloader, calculate_score_grad, calculate_score_grad_vr, constrain_score_by_whole, assign_learning_rate#, data_preparation_pdd

from pdd import PDD
from trainer import PDDTrainer
import random
import numpy as np
import torch
from tqdm import tqdm
import json

def run_single_model(config):
    # configurations initialization
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    dataset_distillation = config['dataset_distillation']
    if dataset_distillation:
        init_logger(config, distill=True, random_or=False, distill_rate=config['dd_rate'])
    else:
        init_logger(config)
    logger = getLogger()
    logger.info(config)
    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data, train_sampler, valid_sampler, test_sampler = data_preparation(config, dataset)#data_preparation_pdd(config, dataset) #the interaction becomes a list of tensors of user_ids
    # model loading and initialization
    # import pdb;pdb.set_trace()
    #你需要做的就是在data_preparation里面把dataloader改一下，使得他negative sampling的时候，不再排掉test和validation里面的正sample(这个其实也不需要改，直接用就好了)。然后把在每个batch里只用top 1改成多一点。
    #train_sampler.datasets[0] is the training dataset, have fileds of: user_ids, item_ids, ratings, neg_item_id, also has attribute of train_sampler.user_num and train_sampler.item_num
    
    model = PDD(config, train_data.dataset).to(config['device'])
    logger.info(model)
    # import pdb;pdb.set_trace()
    # trainer loading and initialization
    trainer = PDDTrainer(config, model)

    # model training
    best_valid_score, best_valid_result, model_parameter, pseudo_dataset = trainer.fit(
        train_data, train_sampler, valid_data, saved=True, show_progress=config['show_progress']
    )
    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    # import pdb;pdb.set_trace()
    #看一下pseudo_dataset 是什么东西
    b = json.dumps(pseudo_dataset)
    file_path = 'pseudo-data/'+config['dataset']+'.json'
    f = open(file_path, 'w')
    f.write(b)
    f.close()
    logger.info(set_color('test result', 'yellow') + f'the pseudo data is stored at:{file_path}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp', help='The datasets can be: ml-1m, yelp, amazon-books, gowalla-merged, alibaba.')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    args, _ = parser.parse_known_args()

    # Config files
    args.config_file_list = [
        'properties/overall.yaml',
        'properties/PDD.yaml'
    ]
    if args.dataset in ['ml-1m', 'yelp', 'amazon-books', 'gowalla-merged', 'alibaba', 'ciao', 'douban']:
        args.config_file_list.append(f'properties/{args.dataset}.yaml')
    if args.config is not '':
        args.config_file_list.append(args.config)

    config = Config(
        model=PDD,
        dataset=args.dataset,
        config_file_list=args.config_file_list
    )

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])

    run_single_model(config)