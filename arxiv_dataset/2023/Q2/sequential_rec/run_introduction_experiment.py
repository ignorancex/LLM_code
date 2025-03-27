# -*- coding: utf-8 -*-
# @Time    : 2020/11/5 21:11
# @Author  : Hui Wang

import os
import numpy as np
import random
import torch
import argparse
import time

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset
from trainers import FinetuneTrainer
from models import S3RecModel
from utils import EarlyStopping, get_user_seqs, get_user_seqs_and_sample, get_item2attribute_json, check_path, set_seed

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/introduction/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=0, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    parser.add_argument("--model_name", default='Finetune_sample', type=str)
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
    parser.add_argument("--gpu_id", type=str, default="1", help="gpu_id")

    parser.add_argument("--sample_num", type=int, default=0, help="sample num of rank loss ")
    parser.add_argument("--aggregation", type=str, default="mean", help="loss aggregation way")
    parser.add_argument("--rank_act", type=str, default="softmax", help="rank similarith act")
    parser.add_argument("--isfull", type=int, default=0, help="TODO")
    parser.add_argument("--loss_type", type=str, default='BCE', help="TODO")
    parser.add_argument("--mask_prod", type=float, default=0.2, help="TODO")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'
    args.sample_file = args.data_dir + args.data_name + '_sample.txt'
    item2attribute_file = args.data_dir + args.data_name + '_item2attributes.json'

    user_seq, max_item, sample_seq = \
        get_user_seqs_and_sample(args.data_file, args.sample_file)
    _, _, valid_rating_matrix, test_rating_matrix = \
        get_user_seqs(args.data_file)

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq, test_neg_items=sample_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = SASRecDataset(args, user_seq, test_neg_items=sample_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    def args_change(args, loss_type, sample_num = 0):
        args.loss_type = loss_type
        args.sample_num = sample_num # 
        args.item2attribute = None

    model_args = {
                  'Ori-BCE': lambda: args_change(args, 'BCE', 0),
                  'BCE-10': lambda: args_change(args, 'BCE', 10),
                  'BCE-100': lambda: args_change(args, 'BCE', 100),
                  'BCE-500': lambda: args_change(args, 'BCE', 500),
                  'BCE-1000': lambda: args_change(args, 'BCE', 1000),
                  'BPR-10': lambda: args_change(args, 'BPR', 10),
                  'BPR-100': lambda: args_change(args, 'BPR', 100),
                  'BPR-max-10': lambda: args_change(args, 'BPR-max', 10),
                  'BPR-max-100': lambda: args_change(args, 'BPR-max', 100),
                  'TOP1-10': lambda: args_change(args, 'TOP1', 10),
                  'TOP1-100': lambda: args_change(args, 'TOP1', 100),
                  'TOP1-max-10': lambda: args_change(args, 'TOP1-max', 10),
                  'TOP1-max-100': lambda: args_change(args, 'TOP1-max', 100),
                  'CE': lambda: args_change(args, 'CE'),
                  'CCE': lambda: args_change(args, 'CCE'),
                 }

    model_result = {}

    for model_name in model_args:
        model_args[model_name]()

        # save model args
        if args.loss_type == 'BCE':
            if args.sample_num == 0:
                args_str = f'{args.model_name}-{args.data_name}-{args.ckp}-{args.loss_type}-1'
        else:
            args_str = f'{args.model_name}-{args.data_name}-{args.ckp}-{args.loss_type}-{args.sample_num}'
        args.log_file = os.path.join(args.output_dir, args_str + '.txt')


        # save model
        checkpoint = args_str + '.pt'
        args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

        print(str(args))
        with open(args.log_file, 'a') as f:
            f.write(str(args) + '\n')

        args.item2attribute = item2attribute



        model = S3RecModel(args=args)
        trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,
                                test_dataloader, args)

        if args.do_eval:
            trainer.load(args.checkpoint_path)
            print(f'Load model from {args.checkpoint_path} for test!')
            trainer.args.train_matrix = test_rating_matrix
            sample_info, full_info = trainer.get_metric(full_sort=True, sample_sort=True)

        else:
            pretrained_path = os.path.join(args.output_dir, f'{args.data_name}-epochs-{args.ckp}.pt')
            try:
                trainer.load(pretrained_path)
                print(f'Load Checkpoint From {pretrained_path}!')

            except FileNotFoundError:
                print(f'{pretrained_path} Not Found! The Model is same as SASRec')

            early_stopping = EarlyStopping(args.checkpoint_path, patience=20, verbose=True, mode = 'decrease')
            start_time = time.time()
            for epoch in range(args.epochs):
                post_fix, avg_loss = trainer.train(epoch)

                post_fix['total training time'] =  '{:.4f}'.format(time.time()-start_time)
                post_fix["max mem"] = '{memory:.0f}'.format(memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                if (epoch + 1) % args.log_freq == 0:
                    print(str(post_fix))

                with open(args.log_file, 'a') as f:
                    f.write(str(post_fix) + '\n')

                early_stopping([avg_loss], trainer.model)
                # scores, _ = trainer.valid(epoch, full_sort=False, verbose = True)
                # # evaluate on MRR
                # early_stopping(scores, trainer.model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            trainer.args.train_matrix = test_rating_matrix
            print('---------------Sample 99 results-------------------')
            # load the best model
            trainer.model.load_state_dict(torch.load(args.checkpoint_path))
            # scores, result_info = trainer.test(0, full_sort=False)
            sample_info, full_info = trainer.get_metric(full_sort=True, sample_sort=True)
        print(args_str)
        print('sample metric:', sample_info)
        print('full metric:', full_info)
        with open(args.log_file, 'a') as f:
            f.write(args_str + '\n')
            f.write(sample_info + '\n')
            f.write(full_info + '\n')
        
        model_result[model_name] = model_name + ' sample metric: ' + str(sample_info) + '\n' +\
                                   model_name + ' full metric: ' + str(full_info)  + '\n' + \
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