import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Dictionary, VQAFeatureDataset
from models import models
from train import train
import math
import vqa_utils
from prettytable import PrettyTable
# from models.rubi import RUBiNet
# from criterion.rubi_criterion import RUBiCriterion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/hdd/robik')
    parser.add_argument('--data_set', type=str, required=True)
    parser.add_argument('--results_path', type=str, default=None)

    parser.add_argument('--do_not_normalize_image_feats', action='store_true')

    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--q_emb_dim', type=int, default=1024)
    parser.add_argument('--model', type=str, default='UpDn')
    parser.add_argument('--apply_rubi', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=7, help='random seed')
    parser.add_argument('--answers_available', type=int, default=1, help='Are the answers available?')
    parser.add_argument('--mode', type=str, choices=['train', 'test'],
                        help='Checkpoint must be specified  for test mode', default='train')
    parser.add_argument('--w_emb_size', type=int, required=False, default=None)
    parser.add_argument('--dictionary_file', type=str, required=False, default=None)
    parser.add_argument('--glove_file', type=str, required=False, default=None)


    parser.add_argument('--spatial_feature_type', type=str, default='none')
    parser.add_argument('--spatial_feature_length', default=0, type=int)
    parser.add_argument('--h5_prefix', required=False, default='use_split', choices=['use_split', 'all'])
    parser.add_argument('--num_objects', required=False, type=int)
    parser.add_argument('--feature_subdir', required=False, default='features')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_expt_dir', type=str)
    parser.add_argument('--resume_expt_name', type=str)
    parser.add_argument('--resume_expt_type', type=str, default='latest', choices=['best', 'latest'])

    parser.add_argument('--expt_name', type=str, required=True)

    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_split', type=str, default='val')
    parser.add_argument('--test_does_not_have_answers', action='store_true')
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--question_rnn_type', type=str, default='GRU')

    parser.add_argument('--additive_fusion', action='store_true')
    parser.add_argument('--multiplicative_fusion', action='store_true')
    parser.add_argument('--question_fusion', action='store_true')
    parser.add_argument('--concat_fusion', action='store_true')

    # RAMEN specific arguments
    parser.add_argument('--mmc_nonlinearity', default='Swish')
    parser.add_argument('--disable_early_fusion', action='store_true')
    parser.add_argument('--disable_late_fusion', action='store_true')
    parser.add_argument('--disable_batch_norm_for_late_fusion', action='store_true')
    parser.add_argument('--mmc_connection', default='residual')
    parser.add_argument('--mmc_aggregator_layers', type=int, default=1)
    parser.add_argument('--mmc_aggregator_dim', type=int, default=1024)
    parser.add_argument('--aggregator_dropout', type=float, default=0)
    parser.add_argument('--mmc_sizes', type=int, nargs='+', default=[2048, 2048, 2048, 2048],
                        help='Layer sizes for Multi Modal Core')
    parser.add_argument('--classifier_sizes', type=int, nargs='+', default=[2048])
    parser.add_argument('--classifier_nonlinearity', type=str, default='Swish')
    parser.add_argument('--input_dropout', default=0, type=float)
    parser.add_argument('--mmc_dropout', default=0, type=float)
    parser.add_argument('--question_dropout_before_rnn', default=None, type=float)
    parser.add_argument('--question_dropout_after_rnn', default=None, type=float)
    parser.add_argument('--classifier_dropout', type=float, default=0.5)

    # Transformer specific arguments
    parser.add_argument('--transformer_aggregation', action='store_true')
    parser.add_argument('--ta_ntoken', type=int, default=36)
    parser.add_argument('--ta_ninp', type=int, default=4096)
    parser.add_argument('--ta_nheads', type=int, default=32)
    parser.add_argument('--ta_nhid', type=int, default=1024)
    parser.add_argument('--ta_nencoders', type=int, default=1)
    parser.add_argument('--ta_dropout', type=float, default=0.2)

    # BAN specific arguments
    parser.add_argument('--glimpse', type=int, default=8)

    # RN specific arguments
    parser.add_argument('--interactor_sizes', type=int, nargs='+', default=[512, 512, 512, 512])
    parser.add_argument('--aggregator_sizes', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--optimizer', type=str, default='Adamax')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[])
    parser.add_argument('--words_dropout', type=float, default=0)
    parser.add_argument('--pre_classification_dropout', type=float, default=0)

    args = parser.parse_args()

    args.dataroot = args.data_root
    if args.results_path is None:
        args.results_path = args.dataroot + '_results'
    args.answers_available = bool(args.answers_available)

    # Handle experiment save/resume
    if args.resume_expt_name is not None:
        args.resume = True
    if args.resume_expt_name is None:
        args.resume_expt_name = args.expt_name
    if args.resume_expt_dir is None:
        args.resume_expt_dir = args.results_path
    args.expt_resume_dir = os.path.join(args.resume_expt_dir, args.resume_expt_name)
    args.expt_save_dir = os.path.join(args.results_path, args.expt_name)

    if not os.path.exists(args.expt_save_dir):
        os.makedirs(args.expt_save_dir)

    args.vocab_dir = os.path.join(args.data_root, args.feature_subdir)
    args.feature_dir = os.path.join(args.data_root, args.feature_subdir)
    if 'clevr' in args.data_set.lower():
        args.token_length = 45
        args.regions = 15
    else:
        args.token_length = 14
        args.regions = 36

    if args.dictionary_file is None:
        args.dictionary_file = args.vocab_dir + '/dictionary.pkl'
    if args.glove_file is None:
        args.glove_file = args.vocab_dir + '/glove6b_init_300d.npy'
    return args


def instance_bce_with_logits(logits, labels):
    """
    Computes binary cross entropy loss
    :param logits:
    :param labels:
    :return:
    """
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def load_bottom_up_dictionary(data_root, features_subdir):
    with open(os.path.join(data_root, features_subdir, 'dictionary.json')) as df:
        qn_word_map = json.load(df)
    with open(os.path.join(data_root, features_subdir, 'answer_ix_map.json')) as af:
        answer_ix_map = json.load(af)
    dictionary = [qn_word_map['word_to_ix'], answer_ix_map['answer_to_ix']]
    return dictionary


def train_model():
    if not args.test:
        train_dset = VQAFeatureDataset(args.train_split, dictionary, data_root=args.dataroot, args=args)
    else:
        train_dset = None
    val_dset = VQAFeatureDataset(args.test_split, dictionary, data_root=args.dataroot, args=args)

    args.w_emb_size = val_dset.dictionary.ntoken
    args.num_ans_candidates = val_dset.num_ans_candidates
    args.dictionary = val_dset.dictionary
    args.v_dim = val_dset.v_dim
    model = getattr(models, args.model)(args)

    count_parameters(model)


    if args.apply_rubi:
        rubi = RUBiNet(model, args.num_ans_candidates, {'input_dim': args.q_emb_dim, 'dimensions': [2048, 2048, 3000]})
        if torch.cuda.is_available():
            model = rubi.cuda()
    else:
        if torch.cuda.is_available():
            model = model.cuda()
    print("Our kickass model {}".format(model))

    optimizer = None
    epoch = 0
    best_val_score = 0
    best_epoch = 0

    if args.resume:
        resume_pth = os.path.join(args.expt_resume_dir, '{}-model.pth'.format(args.resume_expt_type))
        print('Resuming from %s ...' % resume_pth)
        model_data = torch.load(resume_pth)
        if list(model_data['model_state_dict'].keys())[0].startswith('module'):
            model = nn.DataParallel(model)
        model.load_state_dict(model_data['model_state_dict'])
        optimizer = getattr(torch.optim, args.optimizer)(filter(lambda p: p.requires_grad, model.parameters()),
                                                         lr=args.lr)
        # optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer.load_state_dict(model_data['optimizer_state_dict'])
        epoch = model_data['epoch'] + 1
        best_val_score = float(model_data['best_val_score'])
        best_epoch = model_data['best_epoch']
        print("Resumed!")

    if not args.test:
        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=16)
    else:
        train_loader = None
    eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=16)

    if args.apply_rubi:
        criterion = RUBiCriterion()
    else:
        criterion = vqa_utils.instance_bce_with_logits
    train(model, train_loader, eval_loader, args.epochs, optimizer, criterion, args, epoch, best_val_score, best_epoch)

    if not args.test:
        train_dset.close_h5_file()
    val_dset.close_h5_file()

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

if __name__ == '__main__':
    args = parse_args()
    print("Running experiment with these parameters:")
    print(json.dumps(vars(args), indent=4, sort_keys=True))

    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(args.dictionary_file)

    batch_size = args.batch_size
    train_model()
