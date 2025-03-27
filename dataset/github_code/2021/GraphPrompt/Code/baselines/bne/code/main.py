import numpy as np
import random
import os
import torch
import logging
import argparse

from classifier import  Biosyn_Classifier, Graphsage_Classifier,CrossEncoder_Classifier,Graph_Classifier, BNE_Classifier

#set up seed         
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# set up logger
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    if not os.path.exists(name):
        os.makedirs(name)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',type=str,default='../data/datasets/cl.obo')
    parser.add_argument('--classifier_name',type=str,default='graphsage')
    parser.add_argument('--use_text_preprocesser',action='store_true',default=False)
    parser.add_argument('--is_unseen',type=int ,default=True)
    parser.add_argument('--stage_1_model_path',type=str,default='../biobert')
    parser.add_argument('--stage_1_exp_path',type=str,default='../exp/cl/stage_1/unseen')
    parser.add_argument('--stage_2_model_path',type=str,default='../exp/cl/checkpoint_9')
    parser.add_argument('--stage_2_exp_path',type=str,default='../exp/cl/stage_2')

    parser.add_argument('--vocab_file',type=str,default='../biobert/vocab.txt')
    parser.add_argument('--initial_sparse_weight',type=float,default=1.)
    parser.add_argument('--bert_ratio',type=float,default=0.5)
    parser.add_argument('--stage_1_lr',type=float,default=1e-5)
    parser.add_argument('--stage_1_weight_decay',type=float,default=0)
    parser.add_argument('--stage_2_lr',type=float,default=1e-2)
    parser.add_argument('--stage_2_weight_decay',type=float,default=0)

    parser.add_argument('--epoch_num',type=int,default=50)
    parser.add_argument('--top_k',type=int,default=10)
    parser.add_argument('--eval_k',type=int,default=10)
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--score_mode',type=str,default='hybrid')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--save_checkpoint_all',action='store_true',default=False)
    parser.add_argument('--emb_type',type=str,default='bert')

    
    args = parser.parse_args()
    args = args.__dict__

    logger = setup_logger(name=args['stage_2_exp_path'][:],log_file=os.path.join(args['stage_2_exp_path'],'log.log'))
    args['logger'] = logger
    
    print("Experiment arguments are: ", args)

    setup_seed(args['seed'])
    
    #b= Graph_Classifier(args)
    if args['classifier_name'] == 'bne':
        b = BNE_Classifier(args)
        b.train()
        b.eval()
    elif args['classifier_name'] == 'graphsage':
        b = Graph_Classifier(args)
    
    
    #b.save_model_satge_1(os.path.join(args['stage_1_exp_path'],'checkpoint'))
    #b.save_model_stage_2(checkpoint_dir=os.path.join(args['stage_2_exp_path'],'checkpoint'))

    b.eval_stage_1(b.queries_valid,epoch=0)
    b.train_stage_1()

