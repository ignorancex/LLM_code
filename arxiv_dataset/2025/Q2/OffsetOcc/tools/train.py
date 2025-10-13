# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Nicola Marinello, 2025
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from offsetocc.registry import MODELS
from offsetocc.utils import setup_cache_size_limit_of_dynamo
from offsetocc.utils import register_all_modules
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--num_threads',
        type=int,
        default=4,
        help='Set pytorch number of threads')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    if args.num_threads != -1:
        torch.set_num_threads(args.num_threads)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # generate a new wandb run id if wandb is enabled
    if 'visualizer' in cfg and 'vis_backends' in cfg.visualizer:
        for i, backend in enumerate(cfg.visualizer.vis_backends):
            if 'WandbVisBackend' in backend['type']:
                import wandb
                wandb_id = wandb.util.generate_id()
                cfg.visualizer.vis_backends[i]['init_kwargs']['id'] = wandb_id

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None

        if wandb_id and osp.exists(osp.join(cfg.work_dir, 'wandb_id')):
            with open(osp.join(cfg.work_dir, 'wandb_id'), 'r') as f:
                wandb_id = f.read().strip()
            for i, backend in enumerate(cfg.visualizer.vis_backends):
                if 'WandbVisBackend' in backend['type']:
                    cfg.visualizer.vis_backends[i]['init_kwargs']['id'] = wandb_id

    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # this file is saved every time, regardless of whether is it is a resume or not
    # training can always be started as a resume even if it is not
    # this might be handy when condor put the job on hold
    # as the job is restarted with the same command line
    if args.local_rank == 0:
        save_file = osp.join(cfg.work_dir, 'wandb_id')
        # TODO: hack - ugly makedir that 'interferes' with mmengine
        os.makedirs(cfg.work_dir, exist_ok=True)
        with open(save_file, 'w') as f:
            f.write(wandb_id)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()

    model = MODELS.build(cfg.model)

    print(model)


# def main():
#
#     args = parse_args()
#
#     # register all modules in mmdet into the registries
#     register_all_modules()
#
#     cfg = Config.fromfile(args.config)
#
#     model = MODELS.build(cfg.model)
#
#     print(model)

if __name__ == '__main__':
    main()
