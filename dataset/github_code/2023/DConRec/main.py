import argparse
from logging import getLogger

from recbole.config import Config
# from configuratorLocal import Config
from recbole.utils import init_seed, set_color
from logger import init_logger
from utils import create_dataset, data_preparation, create_samplers, obtain_mask, get_dataloader, calculate_score_grad, calculate_score_grad_vr, constrain_score_by_whole, assign_learning_rate, pseudo_data_generation, load_pseudo_data, longtail_interaction_compensation#, data_preparation_pdd

from pdd import PDD
from trainer import PDDTrainer
import random
import numpy as np
import torch
from tqdm import tqdm
from recbole.utils import (
    get_tensorboard,
    # WandbLogger,
)

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

    tensorboard = get_tensorboard(logger)
    # wandblogger = WandbLogger(config)
    '''
    while synthesizing the interactions, you need to update the sampler in data_preparation function.
    you need to disassemble the function of data_preparation. 
    concretely, you could refer to the function get_used_ids in class sampler (recbole.sampler.sampler.py)
    to update the used_ids.
    '''
    # dataset splitting
    if not dataset_distillation:
        train_data, valid_data, test_data = data_preparation(config, dataset)#data_preparation_pdd(config, dataset) #the interaction becomes a list of tensors of user_ids
        # model loading and initialization
        # import pdb;pdb.set_trace()
        model = PDD(config, train_data.dataset).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = PDDTrainer(config, model, logger, tensorboard)

        # model training
        best_valid_score, best_valid_result, model_parameter = trainer.fit(
            train_data, valid_data, saved=True, show_progress=config['show_progress']
        )
        # model evaluation
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')
    else:
        # model_type = config['MODEL_TYPE']
        
        built_datasets = dataset.build()
        # logger = getLogger()

        train_dataset, valid_dataset, test_dataset = built_datasets
        train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)
        # here we need to design the probability score and its update rules
        scores = []
        #augmentation: extend the list of selectable items for dataset distillation
        
        train_inter_user = train_dataset.inter_feat['user_id']
        train_inter_item = train_dataset.inter_feat['item_id']
        
        long_tail_users, long_tail_items = longtail_interaction_compensation(train_inter_user, train_inter_item, config['threshold_tail'])
        if (config['dataset'] != 'douban') and (config['dataset'] != 'gowalla-merged') and (config['dataset'] != 'alibaba') and (config['dataset'] != 'dianping'):
                tail_num = long_tail_items.numpy().shape[0]
                tail_ratings = np.random.choice([3,4,5], tail_num)
                long_tail_ratings = torch.from_numpy(tail_ratings)


        if (config['dataset'] != 'douban') and (config['dataset'] != 'gowalla-merged') and (config['dataset'] != 'alibaba') and (config['dataset'] != 'dianping'):
            train_inter_rating = train_dataset.inter_feat['rating']
        train_inter_num = train_inter_user.shape[0]
        if config['pseudo']:
            logger.info(set_color('Begin Generating Pseudo Data','pink'))
            pseudo_data = pseudo_data_generation(config, train_dataset, valid_dataset, train_sampler, valid_sampler, logger)
            # pseudo_data = load_pseudo_data(config['dataset'])
            sampled_users = np.array(pseudo_data['user_id'])
            sampled_items = np.array(pseudo_data['item_id']) 
            sample_num = len(sampled_items)
            if (config['dataset'] != 'douban') and (config['dataset'] != 'gowalla-merged') and (config['dataset'] != 'alibaba') and (config['dataset'] != 'dianping'):
                sampled_ratings = np.random.choice([3,4,5], sample_num)
        else:
            sample_num = int(train_inter_num * config['supplement_ratio'])
            sampled_items = np.random.choice([i for i in range(1,train_sampler.item_num)], sample_num)
            if (config['dataset'] != 'douban') and (config['dataset'] != 'gowalla-merged') and (config['dataset'] != 'alibaba') and (config['dataset'] != 'dianping'):
                sampled_ratings = np.random.choice([3,4,5], sample_num)
            sampled_users = np.random.choice(list(train_inter_user), sample_num, replace=False)

        ###############
        # if config['cond_learning_rate']:
        config['learning_rate'] = config['condensed_learning_rate']

        #float(1.0/sample_num)
        scores = torch.full((sample_num+train_inter_num,), config['dd_rate']/(1+sample_num/train_inter_num), dtype=torch.float, requires_grad=True, device=config['device'])
        scores.grad = torch.zeros_like(scores)
        logger.info(set_color('Finished augmenting interactions','pink'))

        #接下来就是把这些没见过的items加入sample的used_ids里，可以当作第二版本，如果效果不好再拆出来加进去.
        #直接更新train_dataset就行
        new_train_inter_user = torch.cat((train_inter_user,torch.from_numpy(sampled_users)), 0)#.to(config['device'])
        new_train_inter_item = torch.cat((train_inter_item, torch.from_numpy(sampled_items)), 0)#.to(config['device'])
        if (config['dataset'] != 'douban') and (config['dataset'] != 'gowalla-merged') and (config['dataset'] != 'alibaba') and (config['dataset'] != 'dianping'):
            new_train_ratings = torch.cat((train_inter_rating, torch.from_numpy(sampled_ratings)), 0)#.to(config['device'])

        with torch.no_grad():
            scores_previous = scores 
        outer_lr = config['outer_lr']
        scores_opt = torch.optim.Adam([scores], lr=outer_lr) 
        scores_history = []
        scores_change_record = []
        masks_test = []
        for i in range(0, config['outer_epochs']+1):
            scores_history.append([scores,scores.grad])
            # continue
            model_loss_list_K = []
            derivations_bernoulli_list_K = []
            models = []
            loss_avg = 0
            logger.info(set_color('Outer Loop', 'red') + f': {i}-th iteration:')
            test_results = []
            valid_results = []
            
            for k in range(0,config['expectation_num_k']):
                mask, grad_derivation_bernoulli = obtain_mask(scores)
                derivations_bernoulli_list_K.append(grad_derivation_bernoulli)
                
                nonZeroIndex = torch.abs(mask)>1e-3
                synthesized_train_user = new_train_inter_user[nonZeroIndex]
                synthesized_train_item = new_train_inter_item[nonZeroIndex]
                if (config['dataset'] != 'douban') and (config['dataset'] != 'gowalla-merged') and (config['dataset'] != 'alibaba') and (config['dataset'] != 'dianping'):
                    synthesized_train_rating = new_train_ratings[nonZeroIndex]

                compensated_train_user = torch.cat((synthesized_train_user, long_tail_users),0)
                compensated_train_item = torch.cat((synthesized_train_item, long_tail_items),0)
                train_dataset.inter_feat.interaction.update({'user_id':compensated_train_user})
                train_dataset.inter_feat.interaction.update({'item_id':compensated_train_item})
                # import pdb;pdb.set_trace()
                ####################################################################
                if (config['dataset'] != 'douban') and (config['dataset'] != 'gowalla-merged') and (config['dataset'] != 'alibaba') and (config['dataset'] != 'dianping'):
                    compensated_train_rating = torch.cat((synthesized_train_rating,long_tail_ratings),0)
                    train_dataset.inter_feat.interaction.update({'rating':compensated_train_rating})
                train_dataset.inter_feat.length = train_dataset.inter_feat['user_id'].shape[0]

                train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
                valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)
                test_data = get_dataloader(config, 'evaluation')(config, test_dataset, test_sampler, shuffle=False)
                test_data_for_loss = get_dataloader(config, 'train')(config, valid_dataset, test_sampler, shuffle=False)

                model = PDD(config, train_data.dataset).to(config['device'])
                trainer = PDDTrainer(config, model, logger, tensorboard)
                try:
                    if config['convergence_inner']: #or i == config['outer_epochs']:
                        # if i == config['outer_epochs'] - 1:
                        #     logger.info(set_color(f'Outer Iteration:{i}-th, K:{k}, best valid ',
                        #                           'yellow') + f': {valid_results[k]}')
                        # import pdb;pdb.set_trace()
                        best_valid_score, best_valid_result, model_k = trainer.fit(
                            train_data, valid_data, saved=True, show_progress=config['show_progress']
                        )
                    else:
                        model_k = trainer.fit_steps(
                            train_data, valid_data, saved=True, show_progress=config['show_progress'], inner_epochs = config['inner_epochs']
                        )
                except TypeError as e:
                    print('TypeError!!! ', e)
                    import pdb;pdb.set_trace()
                    best_valid_score, best_valid_result, model_k = trainer.fit(
                        train_data, valid_data, saved=True, show_progress=config['show_progress']
                    )
                # model evaluation
                # if config['convergence_inner']:
                #     models.append(model_k)
                #     test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
                #     test_results.append(test_result)
                #     logger.info(set_color(f'Outer Iteration:{i}-th, K:{k}, best valid ', 'yellow') + f': {best_valid_result}')
                #     logger.info(set_color(f'Outer Iteration:{i}-th, K:{k}, test result', 'yellow') + f': {test_result}')
                if i == config['outer_epochs']:
                    # test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
                    # test_results.append(test_result)
                    # valid_results.append(best_valid_result)
                    masks_test.append(mask)
                else:
                    models.append(model_k)
                
                torch.cuda.empty_cache()

            if i == config['outer_epochs']:
                # for k in range(0,config['expectation_num_k']):
                #     logger.info(set_color(f'Outer Iteration:{i}-th, K:{k}, best valid ', 'yellow') + f': {valid_results[k]}')
                #     logger.info(set_color(f'Outer Iteration:{i}-th, K:{k}, test result', 'yellow') + f': {test_results[k]}')
                continue
            else:#update the scores and matrix
                for k in range(0, config['expectation_num_k']):
                    loss_k = trainer.valid_loss_steps(test_data_for_loss, models[k], i, show_progress=config['show_progress'])
                    if config['model_selection'] == 'XSimGCL':
                        loss_k = sum(list(loss_k))
                    loss_avg += loss_k/config['expectation_num_k']
                    model_loss_list_K.append(loss_k)
                with torch.no_grad():
                    if config['vr']:
                        calculate_score_grad_vr(scores, model_loss_list_K, loss_avg, derivations_bernoulli_list_K, config['expectation_num_k'])
                    else:
                        calculate_score_grad(scores, model_loss_list_K, derivations_bernoulli_list_K, config['expectation_num_k'])
                #先不用assign learning rate
                if config['outer_lr_decay'] and i%config['outer_lr_decay_per_epochs']==0 and outer_lr >=config['outer_lr_decay_lower_bound']:#1e-7:
                    outer_lr = outer_lr /config['outer_lr_decay_rate']
                assign_learning_rate(scores_opt, outer_lr)
                scores_opt.step()
                constrain_score_by_whole(scores, config['dd_rate'])

        test_results = []
        valid_results = []
        logger.info(set_color(f'Start evaluating!!!', 'Red'))
        for k in range(0,config['expectation_num_k']):
            nonZeroIndex = torch.abs(masks_test[k])>1e-3
            synthesized_train_user = new_train_inter_user[nonZeroIndex]
            synthesized_train_item = new_train_inter_item[nonZeroIndex]
            if (config['dataset'] != 'douban') and (config['dataset'] != 'gowalla-merged') and (config['dataset'] != 'alibaba') and (config['dataset'] != 'dianping'):
                synthesized_train_rating = new_train_ratings[nonZeroIndex]

            compensated_train_user = torch.cat((synthesized_train_user, long_tail_users),0)
            compensated_train_item = torch.cat((synthesized_train_item, long_tail_items),0)
            train_dataset.inter_feat.interaction.update({'user_id':compensated_train_user})
            train_dataset.inter_feat.interaction.update({'item_id':compensated_train_item})

            ####################################################################
            if (config['dataset'] != 'douban') and (config['dataset'] != 'gowalla-merged') and (config['dataset'] != 'alibaba') and (config['dataset'] != 'dianping'):
                compensated_train_rating = torch.cat((synthesized_train_rating,long_tail_ratings),0)
                train_dataset.inter_feat.interaction.update({'rating':compensated_train_rating})
            train_dataset.inter_feat.length = train_dataset.inter_feat['user_id'].shape[0]

            train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
            valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)
            test_data = get_dataloader(config, 'evaluation')(config, test_dataset, test_sampler, shuffle=False)
            test_data_for_loss = get_dataloader(config, 'train')(config, valid_dataset, test_sampler, shuffle=False)
            
            config['model_selection'] = config['test_model_selection']
            config['ssl_or_not'] = config['test_model_ssl']
            if config['test_model_selection'] == 'XSimGCL':
                config['ssl_or_not'] = True
            config['noise_eps'] = config['test_noise_eps']
            config['learning_rate'] = config['test_model_learning_rate']

            model = PDD(config, train_data.dataset).to(config['device'])
            trainer = PDDTrainer(config, model, logger, tensorboard)
            best_valid_score, best_valid_result, model_k = trainer.fit(
                train_data, valid_data, saved=True, show_progress=config['show_progress']
            ) 
            test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
            valid_results.append(best_valid_result)
            test_results.append(test_result)
        for k in range(0,config['expectation_num_k']):
            logger.info(set_color(f'Here are the final results!!!', 'Red'))
            logger.info(set_color(f'K:{k}, best valid ', 'yellow') + f': {valid_results[k]}')
            logger.info(set_color(f'K:{k}, test result', 'yellow') + f': {test_results[k]}')
        
            
        print('Finished Dataset Distillation!')
        logger.info(set_color('Finished Dataset Distillation! The scores probabilities are:','red'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp', help='The datasets can be: ml-1m, yelp, amazon-books, gowalla-merged, alibaba.')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    parser.add_argument('--test_model_selection', type=str, default='BPRMF', help='test_model_selection.')
    parser.add_argument('--model_selection', type=str, default='BPRMF', help='model_selection.')
    # parser.add_argument('--gpu_id', type=int, default=3, help='gpu_id.')

    args, _ = parser.parse_known_args()

    # Config files
    args.config_file_list = [
        # 'properties/overall.yaml'
        # 'properties/PDD.yaml'
    ]
    test_model_selection = args.test_model_selection
    model_selection = args.model_selection
    # gpu_id = args.gpu_id
    if args.dataset in ['ml-20m','ml-1m', 'yelp', 'amazon-books', 'gowalla-merged', 'alibaba', 'ciao', 'douban']:
        args.config_file_list.append(f'properties/{args.dataset}.yaml')
        args.config_file_list.append(f'properties/PDD-{args.dataset}.yaml')
        args.config_file_list.append(f'properties/overall-{args.dataset}.yaml')
    # import pdb;pdb.set_trace()
    if args.dataset in ['dianping']:
        # args.config_file_list.append(f'properties/{args.dataset}.yaml')
        args.config_file_list.append(f'properties/overall-{args.dataset}.yaml')
        args.config_file_list.append(f'properties/PDD-{args.dataset}.yaml')
    if args.config is not '':
        args.config_file_list.append(args.config)
    # import pdb;pdb.set_trace()
    config = Config(
        model=PDD,
        dataset=args.dataset,
        config_file_list=args.config_file_list#,
        # gpu_id_input = gpu_id
    )
    config['model_selection'] = model_selection
    config['test_model_selection'] = test_model_selection
    # config['gpu_id'] = gpu_id
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu_id}'

    # config['device'] = torch.device(f"cuda") if torch.cuda.is_available() else "cpu"


    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    # import pdb;pdb.set_trace()

    run_single_model(config)