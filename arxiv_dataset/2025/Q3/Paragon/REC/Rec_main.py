import os
import sys
import logging
import configparser
import pandas as pd
import torch
from types import SimpleNamespace

from runner import runner_with_LoRA
from reader import SeqReader
from models import SASRec
from models import GRU4Rec
from models import TiSASRec
from utils import utils

def load_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def config_to_args(config):
    args = SimpleNamespace()
    
    for section in config.sections():
        for key, value in config[section].items():
            setattr(args, key, value)
    
    args.gpu = config['DEFAULT']['gpu']
    args.random_seed = int(config['DEFAULT']['random_seed'])
    args.verbose = int(config['DEFAULT']['verbose'])
    args.load = int(config['DEFAULT']['load'])
    args.train = int(config['DEFAULT']['train'])
    args.save_final_results = int(config['DEFAULT']['save_final_results'])
    args.regenerate = int(config['DEFAULT']['regenerate'])
    
    args.all_epoch = int(config['TRAIN']['all_epoch'])
    args.save_epoch = int(config['TRAIN']['save_epoch'])
    args.pretrain_epochs = int(config['TRAIN']['pretrain_epochs'])
    args.finetune_epochs = int(config['TRAIN']['finetune_epochs'])
    args.check_epoch = int(config['TRAIN']['check_epoch'])
    args.test_epoch = int(config['TRAIN']['test_epoch'])
    args.early_stop = int(config['TRAIN']['early_stop'])
    args.lr = float(config['TRAIN']['lr'])
    args.l2 = float(config['TRAIN']['l2'])
    args.batch_size = int(config['TRAIN']['batch_size'])
    args.eval_batch_size = int(config['TRAIN']['eval_batch_size'])
    args.num_workers = int(config['TRAIN']['num_workers'])
    args.pin_memory = int(config['TRAIN']['pin_memory'])
    
    args.emb_size = int(config['MODEL']['emb_size'])
    args.history_max = int(config['MODEL']['history_max'])
    args.num_neg = int(config['MODEL']['num_neg'])
    args.dropout = float(config['MODEL']['dropout'])
    args.test_all = int(config['MODEL']['test_all'])
    args.model_path_root = f"./checkpoints/{args.model_name}_{args.dataset}/{args.model_name}_acc{args.finetune_acc_weight}_div{args.finetune_div_weight}/"
    args.buffer = 1

    # Model-specific parameters
    if args.model_name in ['SASRec', 'TiSASRec']:
        args.num_layers = int(config['MODEL']['num_layers'])
        args.num_heads = int(config['MODEL']['num_heads'])
    
    if args.model_name == 'TiSASRec':
        args.time_max = int(config['MODEL']['time_max'])
    
    if args.model_name == 'GRU4Rec':
        args.hidden_size = int(config['MODEL']['hidden_size'])
    
    args.accuracy_weight = float(config['EVALUATION']['accuracy_weight'])
    args.diversity_weight = float(config['EVALUATION']['diversity_weight'])
    args.finetune_acc_weight = float(config['EVALUATION']['finetune_acc_weight'])
    args.finetune_div_weight = float(config['EVALUATION']['finetune_div_weight'])
    args.test_sample_ratio = float(config['EVALUATION']['test_sample_ratio'])
    
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    
    return args

def setup_logging(args):
    if not args.log_file and args.mode == 'train':
        log_args = [args.model_name, args.dataset, str(args.random_seed)]
        print(f'-----{args.model_name}')
        log_file_name = '__'.join(log_args).replace(' ', '__')
        args.log_file = f"{args.model_path_root}{log_file_name}.log"
    if not args.log_file and args.mode == 'test':
        args.log_file = f"./log/{args.test_model_path.replace('/' , '_')}.log"
    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def create_model(args, corpus):
    if args.model_name == 'SASRec':
        return SASRec.SASRec(args, corpus).to(args.device)
    if args.model_name == 'GRU4Rec':
        return GRU4Rec.GRU4Rec(args, corpus).to(args.device)
    if args.model_name == 'TiSASRec':
        return TiSASRec.TiSASRec(args, corpus).to(args.device)
    else:
        raise ValueError(f"Unknown model type: {args.model_name}")

def save_rec_results(dataset, runner, topk, args):
    model_name = f"{args.model_name}"
    result_path = os.path.join(runner.log_path, runner.save_appendix, f'rec-{model_name}-{dataset.phase}.csv')
    utils.check_dir(result_path)
    logging.info('Saving top-{} recommendation results to: {}'.format(topk, result_path))
    out_dict = runner.predict(dataset)
    predictions = out_dict['predictions']
    users, rec_items, rec_predictions = list(), list(), list()
    for i in range(len(dataset)):
        info = dataset[i]
        users.append(info['user_id'])
        item_scores = zip(info['item_id'], predictions[i])
        sorted_lst = sorted(item_scores, key=lambda x: x[1], reverse=True)[:topk]
        rec_items.append([x[0] for x in sorted_lst])
        rec_predictions.append([x[1] for x in sorted_lst])
    rec_df = pd.DataFrame(columns=['user_id', 'rec_items', 'rec_predictions'])
    rec_df['user_id'] = users
    rec_df['rec_items'] = rec_items
    rec_df['rec_predictions'] = rec_predictions
    rec_df.to_csv(result_path, sep='\t', index=False)
    logging.info("{} Prediction results saved!".format(dataset.phase))

def main(args):
    utils.init_seed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logging.info(f'Device: {args.device}')

    # Load corpus
    corpus_path = os.path.join(args.path, args.dataset, f"{SASRec.SASRec.reader}.pkl")
    if not args.regenerate and os.path.exists(corpus_path):
        logging.info(f'Load corpus from {corpus_path}')
        corpus = pd.read_pickle(corpus_path)
    else:
        corpus = SeqReader.SeqReader(args)
        logging.info(f'Save corpus to {corpus_path}')
        corpus.to_pickle(corpus_path)

    model = create_model(args, corpus)
    logging.info(f'#params: {model.count_variables()}')
    logging.info(model)

    if args.mode == 'train':
        data_dict = {}
        for phase in ['train', 'dev']:
            data_dict[phase] = model.Dataset(model, corpus, phase)
            data_dict[phase].prepare()
        data_dict['test'] = model.Dataset(model, corpus, 'test', sample_ratio=args.test_sample_ratio)
        data_dict['test'].prepare()

        runner = runner_with_LoRA.BaseRunner(args)
        if args.load > 0:
            model.load_model()
        if args.train > 0:
            runner.train(data_dict)

        if args.save_final_results == 1:
            save_rec_results(data_dict['dev'], runner, 100, args)
            save_rec_results(data_dict['test'], runner, 100, args)
        model.actions_after_train()
        logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)

    if args.mode == 'test':
        result_dev = []
        path = args.test_model_path
        new_state_dict = torch.load(path)
        model.load_state_dict(new_state_dict)

        data_dict = {'dev': model.Dataset(model, corpus, 'dev')}
        data_dict['dev'].prepare()

        runner = runner_with_LoRA.BaseRunner(args)
        eval_res = runner.easy_print_res(data_dict['dev'])
        result_dev.append(list(eval_res.values()))

        logging.info(f'\nDev After Training: {eval_res}')

if __name__ == '__main__':
    config = load_config()
    args = config_to_args(config)
    setup_logging(args)
    main(args)