# -*- coding: utf-8 -*-
# @Time    : 2020/4/25 22:59
# @Author  : Hui Wang

import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset
from trainers import FinetuneTrainer
from models import S3RecModel, GRU4Rec, SRGNN
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/table2_results/', type=str)
    parser.add_argument('--data_name', default='LastFM', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=0, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    parser.add_argument("--model_name", default='SASRec', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # our add params
    parser.add_argument("--sample_num", type=int, default=0, help="sample num of rank loss ")
    parser.add_argument("--aggregation", type=str, default="mean", help="loss aggregation way")
    parser.add_argument("--rank_act", type=str, default="softmax", help="rank similarith act")
    parser.add_argument("--isfull", type=int, default=0, help="TODO")
    parser.add_argument("--loss_type", type=str, default='BCE', help="TODO")
    parser.add_argument("--mask_prod", type=float, default=0.2, help="TODO")
    parser.add_argument("--step", type=int, default=1, help="GNN Cell step")

    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir += f'{args.data_name}/'
    check_path(args.output_dir)


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'
    item2attribute_file = args.data_dir + args.data_name + '_item2attributes.json'

    user_seq, max_item, valid_rating_matrix, test_rating_matrix = \
        get_user_seqs(args.data_file)

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.ckp}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    # check_path(args.output_dir + f'{args.data_name}/')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # tensorboard 
    args.writer = None

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = SASRecDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)


    dataset2ckp = {'LastFM': 150,
                   'Beauty': 150,
                   'Sports_and_Outdoors': 100,
                   'Toys_and_Games': 150,
                   'Yelp': 100,
                  }

    def args_change(args, loss_type, model_name):
        args.loss_type = loss_type
        args.item2attribute = None
        args.model_name = model_name
        args.train_matrix = valid_rating_matrix
        if model_name == 'S3Rec':
            args.ckp = dataset2ckp[args.data_name]
        else:
            args.ckp = 0

    model_args = {
                  'Ori-GRU4Rec': lambda: args_change(args, 'CE', 'GRU4Rec'), 
                  'Our-GRU4Rec': lambda: args_change(args, 'CCE', 'GRU4Rec'), 
                  'Ori-SRGNN': lambda: args_change(args, 'CE', 'Ori-SRGNN'),
                  'Our-SRGNN': lambda: args_change(args, 'CCE', 'SRGNN'),
                  'Ori-SASRec': lambda: args_change(args, 'BCE', 'SASRec'), 
                  'Our-SASRec': lambda: args_change(args, 'CCE', 'SASRec'), 
                  'Ori-S3Rec': lambda: args_change(args, 'BCE', 'S3Rec'), 
                  'Our-S3Rec': lambda: args_change(args, 'CCE', 'S3Rec'), 
                 }

    model_result = {}

    for model_name in model_args:
        model_args[model_name]()

        args_str = f'{args.model_name}-{args.data_name}-{args.ckp}-{args.loss_type}'
        args.log_file = os.path.join(args.output_dir, args_str + '.txt')

        checkpoint = args_str + '.pt'
        args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

        if 'GRU4Rec' == args.model_name:
            model = GRU4Rec(args=args)
        elif 'SRGNN' in args.model_name:
            model = SRGNN(args = args)
        else:
            model = S3RecModel(args=args)

        trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,
                                test_dataloader, args)

        if args.do_eval:
            trainer.load(args.checkpoint_path)
            print(f'Load model from {args.checkpoint_path} for test!')
            scores, result_info = trainer.test(0, full_sort=True)

        else:
            pretrained_path = os.path.join('reproduce/', f'{args.data_name}-epochs-{args.ckp}.pt')
            try:
                trainer.load(pretrained_path)
                print(f'Load Checkpoint From {pretrained_path}!')

            except FileNotFoundError:
                print(f'{pretrained_path} Not Found! The Model is same as SASRec')

            early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
            for epoch in range(args.epochs):
                post_fix = trainer.train(epoch)
                # evaluate on NDCG@20
                scores, _ = trainer.valid(epoch, full_sort=True, verbose=True)
                early_stopping(np.array(scores[-1:]), trainer.model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            trainer.args.train_matrix = test_rating_matrix
            print('---------------Change to test_rating_matrix!-------------------')
            # load the best model
            trainer.model.load_state_dict(torch.load(args.checkpoint_path))
            _, full_info = trainer.test(epoch, full_sort=True, verbose=True)
        print(args_str)
        print('full metric:', full_info)
        with open(args.log_file, 'a') as f:
            f.write(args_str + '\n')
            # f.write(sample_info + '\n')
            f.write(full_info + '\n')
        
        model_result[model_name] = model_name + ' full metric: ' + str(full_info)  + '\n' + \
                                   str(post_fix)
        
        del(model)
        del(trainer)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

        final_result_file = os.path.join(args.output_dir, 'All_result' + '.txt')
        with open(final_result_file, 'a') as f:
            f.write(model_result[model_name] + '\n')
            f.write('\n')


main()