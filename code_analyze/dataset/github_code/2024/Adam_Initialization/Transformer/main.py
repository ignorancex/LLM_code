#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import os
print('pid:', os.getpid())
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import random
import json
import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, progress_bar, tasks, utils # options
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from scripts.average_checkpoints import main_average_ckpt
from generate import main as generate_main

import options
from optimizers.optim_utils import init_optim
from optimizers.logger import Logger

fb_pathmgr_registerd = False

OVERRIDE_ARGS=True

def add_optim_init_parser(parser):
    group = parser.add_argument_group('Optim Initialization')
    # fmt: off
    #group.add_argument('data', default="./data-bin/iwslt14.tokenized.de-en.joined", type=str, )
    parser.add_argument('--dataset', default='iwslt14.de-en', type=str,
                        help='dataset name')

    parser.add_argument("--init_method", default='random', type=str,
                        choices=['none','random','random-kaiming','grad-mean','grad-sq','grad-var','grad-mean-var','grad-mean-random'])
    parser.add_argument('--init_state', default="mv", type=str,choices=['mv','m','v'])
    parser.add_argument('--init_scale', default=1.0, type=float,)
    parser.add_argument('--init_scale_m0', default=1.0, type=float,)
    parser.add_argument('--init_size', default=5000, type=int,)

    parser.add_argument('--save_name', default="result_new", type=str,)

    parser.add_argument('--optim_default_args', default=1, type=int,)

    group.add_argument('--use_warmup', default=1, type=int, )

    parser.add_argument('--num-epoch-checkpoints', default=5, type=int,
                        help='if set, will try to find checkpoints with names checkpoint_xx.pt in the path specified by input, '
                             'and average last this many of them.')
    parser.add_argument('--checkpoint-upper-bound', type=int,
                        help='when using --num-epoch-checkpoints, this will set an upper bound on which checkpoint to use, '
                             'e.g., with --num-epoch-checkpoints=10 --checkpoint-upper-bound=50, checkpoints 41-50 would be averaged.')


def main(args, init_distributed=False):
    utils.import_user_module(args)

    try:
        from fairseq.fb_pathmgr import fb_pathmgr
        global fb_pathmgr_registerd
        if not fb_pathmgr_registerd:
            fb_pathmgr.register()
            fb_pathmgr_registerd = True
    except (ModuleNotFoundError, ImportError):
        pass

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))
    # filter the params that is unused for finetuing, ad-hoc for finetuing, should turn off when bert pretraining.
    for n, p in model.named_parameters():
       if "lm_head" in n:
           p.requires_grad = False
        #    print(n)
    #    print(n, p.requires_grad, p.shape)
    # for i, (n, p) in enumerate(model.named_parameters()):
        # print(i, n, p.size())
    # asdf

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')

    if not hasattr(checkpoint_utils.save_checkpoint, 'not_best'):
        checkpoint_utils.save_checkpoint.not_best = 0


    if args.init_method !='none':
        print('optim init')
        update_freq = args.update_freq[epoch_itr.epoch - 1] \
            if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

        # Initialize data iterator
        itr = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=args.fix_batches_to_gpus,
            shuffle=(epoch_itr.epoch >= args.curriculum),
        )
        iterator = iterators.GroupedIterator(itr, update_freq)
        iterator = progress_bar.build_progress_bar(
            args, iterator, epoch_itr.epoch, no_progress_bar='simple',
        )

        scale_logger = init_optim(trainer.optimizer.optimizer, criterion=criterion, data_set=None, model=model,
                   init_method=args.init_method, init_state=args.init_state, scaling_factor=args.init_scale,
                   scaling_factor_m0=args.init_scale_m0, device='cuda:0', data_size=args.init_size,
                   algo=args.optimizer,trainer=trainer, progress=iterator)

    logger = Logger( args.save_log_path +'_log.txt', title='tranformer_nmt')
    logger.set_names(['Epoch', 'Train loss', 'Test loss', 'Train PPL', 'Test PPL', 'Grad Norm', 'LR'])
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            if args.early_stop > 0:
                if hasattr(checkpoint_utils.save_checkpoint, 'best') and valid_losses[0] > checkpoint_utils.save_checkpoint.best:
                    checkpoint_utils.save_checkpoint.not_best += 1
                    print("| Not the best ckpt... not best:", checkpoint_utils.save_checkpoint.not_best)
                    if checkpoint_utils.save_checkpoint.not_best > args.early_stop:
                        print("| Early stop...")
                        break
                else:
                    checkpoint_utils.save_checkpoint.not_best = 0
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        reload_dataset = ':' in getattr(args, 'data', '')
        # sharded data: get train iterator for next epoch
        epoch_itr = trainer.get_train_iterator(epoch_itr.epoch, load_dataset=reload_dataset)

        train_stats = get_training_stats(trainer)
        valid_stats = get_valid_stats(trainer, args)
        logger.append_list([epoch_itr.epoch, train_stats['loss'].avg, valid_stats['loss'].avg, train_stats['ppl'], valid_stats['ppl'], train_stats['gnorm'].avg, lr])

    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))
    logger.print_summary()
    logger.close()

    return


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second and updates-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()
            trainer.get_meter('ups').reset()

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer, args, extra_meters)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(
            stats[args.best_checkpoint_metric].avg
            if args.best_checkpoint_metric == 'loss'
            else stats[args.best_checkpoint_metric]
        )
    return valid_losses


def get_valid_stats(trainer, args, extra_meters=None):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min

        current_metric = None
        if args.best_checkpoint_metric == 'loss':
            current_metric = stats['loss'].avg
        elif args.best_checkpoint_metric in extra_meters:
            current_metric = extra_meters[args.best_checkpoint_metric].avg
        elif args.best_checkpoint_metric in stats:
            current_metric = stats[args.best_checkpoint_metric]
        else:
            raise ValueError("best_checkpoint_metric not found in logs")

        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            current_metric,
        )
    return stats


def parse_bleu_string(bleu_str):
    # Split the string by the equal sign and commas
    parts = bleu_str.split(", ")

    # Initialize dictionary to hold the results
    bleu_dict = {}

    # The first part is the BLEU4 score
    bleu4_part = parts[0].split(" = ")
    bleu_dict["BLEU4"] = float(bleu4_part[1])

    # The second part is the 0.0/0.0/0.0/0.0 scores
    precision_scores = parts[1].split("/")
    bleu_dict["precisions"] = [float(score) for score in precision_scores]

    # The rest are key-value pairs like BP, ratio, syslen, and reflen
    for part in parts[2:]:
        key, value = part.split("=")
        try:
            bleu_dict[key] = float(value)
        except ValueError:
            bleu_dict[key] = int(value)

    return bleu_dict

def eval_model(args):
    print('begin to eval BLUE score')
    main_average_ckpt(args)
    args.path = args.output_ckpt
    args.batch_size= 128
    args.remove_bpe = '@@ '
    args.log_format = 'simple'
    args.source_lang = 'de'
    args.target_lang = 'en'
    args.results_path = os.path.join(args.save_dir,'gen_result')
    os.makedirs(args.results_path)
    scorer = generate_main(args)
    print('generate translation done')
    # with open(os.path.join(args.results_path, 'generate-test.txt'), 'r',  encoding='utf-8') as h:
    #     texts=[x.strip() for x in h.readlines()]
    bleu_str =scorer.result_string() # texts[-1].split(':')[1].split(',')[0].replace(' ','')
    with open(os.path.join(args.save_dir, 'bleu_score.txt'), 'w',  encoding='utf-8') as h:
        h.write(bleu_str)
    bleu_str_cleaned = bleu_str.replace("(", "").replace(")", "")  # Remove the parentheses
    bleu_dict = parse_bleu_string(bleu_str_cleaned)

    with open(args.save_log_path+'_bleu.json', 'w') as json_file:
        json.dump(bleu_dict, json_file)

    print('Eval done.')
    print(bleu_dict)

    return bleu_dict

def run_main():

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    parser = options.get_training_parser(default_task="translation_custom")
    options.add_generation_args(parser)
    add_optim_init_parser(parser)
    args = options.parse_args_and_arch(parser)
    try:
        adam_eps = args.adam_eps
    except:
        adam_eps = 0.0
    os.makedirs(args.save_name, exist_ok=True)
    os.makedirs("checkpoints"+'_'+args.save_name, exist_ok=True)
    args.save_flag = f"transformer_opt{args.optimizer}-{args.lr[0]}-eps{adam_eps}_init{args.init_method}-{args.init_state}-{args.init_scale}-{args.init_scale_m0}-{args.init_size}_seed{args.seed}_warm{args.use_warmup}"

    args.save_log_path = os.path.join(args.save_name, args.save_flag)
    args.save_dir = os.path.join('checkpoints'+'_'+args.save_name, args.save_flag)
    args.input_ckpts = [args.save_dir]
    args.output_ckpt = os.path.join(args.save_dir, 'ckpt_avg.pt')

    if not args.use_warmup:
        args.warmup_init_lr = args.lr[0]
        print(f'did not use warmup, lr={args.warmup_init_lr}~{args.lr[0]}')
    else:
        print(f'use warmup, lr={args.warmup_init_lr}~{args.lr[0]}')

    main(args)

    eval_model(args)



if __name__ == '__main__':
    run_main()
