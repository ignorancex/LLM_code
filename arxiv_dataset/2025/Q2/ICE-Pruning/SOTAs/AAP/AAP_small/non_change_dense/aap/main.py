import argparse
import os
import shutil
import time
import random
import sys

import numpy as np

import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from image_classification.training import *
from image_classification.utils import *

import dllogger

from image_classification import pruning_utils, dataloader

np.seterr(all="ignore")

def add_parser_arguments(parser):

    parser.add_argument("--data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "--pruning_rate",
        metavar="p",
        default=0.3,
        type=float,
        help="number of classes in the dataset",
    )
 
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet50",
    )

    parser.add_argument(
        "--model-config",
        "-c",
        metavar="CONF",
        default="classic",
    )

    parser.add_argument(
        "--num-classes",
        metavar="N",
        default=1000,
        type=int,
        help="number of classes in the dataset",
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=5,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 5)",
    )

    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "--run-epochs",
        default=-1,
        type=int,
        metavar="N",
        help="run only N epochs, used for checkpointing runs",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256) per gpu",
    )

    parser.add_argument(
        "--optimizer-batch-size",
        default=-1,
        type=int,
        metavar="N",
        help="size of a total batch size, for simulating bigger batches using gradient accumulation",
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )

    parser.add_argument(
        "--lr-schedule",
        default="step",
        type=str,
        metavar="SCHEDULE",
        choices=["step", "linear", "cosine", "cifar10_rn18_step", "cifar10_vgg_step"],
        help="Type of LR schedule: {}, {}, {}".format("step", "linear", "cosine"),
    )

    parser.add_argument(
        "--warmup", default=0, type=int, metavar="E", help="number of warmup epochs"
    )

    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        metavar="S",
        help="label smoothing",
    )

    parser.add_argument(
        "--mixup", default=0.0, type=float, metavar="ALPHA", help="mixup alpha"
    )

    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )

    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )

    parser.add_argument(
        "--bn-weight-decay",
        action="store_true",
        help="use weight_decay on batch normalization learnable parameters, (default: false)",
    )

    parser.add_argument(
        "--nesterov",
        action="store_true",
        help="use nesterov momentum, (default: false)",
    )

    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )

    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )

    parser.add_argument(
        "--pretrained-weights",
        default="",
        type=str,
        metavar="PATH",
        help="load weights from here",
    )

    parser.add_argument("--fp16", action="store_true", help="Run model fp16 mode.")
    parser.add_argument(
        "--static-loss-scale",
        type=float,
        default=1,
        help="Static loss scale, positive power of 2 values can improve fp16 convergence.",
    )
    parser.add_argument(
        "--dynamic-loss-scale",
        action="store_true",
        help="Use dynamic loss scaling.  If supplied, this argument supersedes "
        + "--static-loss-scale.",
    )
    parser.add_argument(
        "--prof", type=int, default=-1, metavar="N", help="Run only N iterations"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run model AMP (automatic mixed precision) mode.",
    )

    parser.add_argument(
        "--seed", default=None, type=int, help="random seed used for numpy and pytorch"
    )

    parser.add_argument(
        "--gather-checkpoints",
        action="store_true",
        help="Gather checkpoints throughout the training, without this flag only best and last checkpoints will be stored",
    )

    parser.add_argument(
        "--raport-file",
        default="experiment_raport.json",
        type=str,
        help="file in which to store JSON experiment raport",
    )

    parser.add_argument(
        "--evaluate", action="store_true", help="evaluate checkpoint/model"
    )
    parser.add_argument("--training-only", action="store_true", help="do not evaluate")

    parser.add_argument(
        "--no-checkpoints",
        action="store_false",
        dest="save_checkpoints",
        help="do not store any checkpoints, useful for benchmarking",
    )

    parser.add_argument("--checkpoint-filename", default="checkpoint.pth.tar", type=str)
    
    parser.add_argument(
        "--workspace",
        type=str,
        default="./",
        metavar="DIR",
        help="path to directory where checkpoints will be stored",
    )
    parser.add_argument(
        "--memory-format",
        type=str,
        default="nchw",
        choices=["nchw", "nhwc"],
        help="memory layout, nchw or nhwc",
    )

    # Pruning Hparams
    parser.add_argument("--prune_type", default="AAP", type=str)
    parser.add_argument("--lr_decay_epochs", default='', type=str)
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='decay rate for learning rate')
    parser.add_argument("--num_pruningRound", default=25, type=int)
    parser.add_argument("--rewind_epoch", default=-1, type=int)
    parser.add_argument("--num_trails", default=1, type=int)
    parser.add_argument("--power_value", default=-1, type=int)
    parser.add_argument("--target_accloss", default=0.0, type=float)
    parser.add_argument("--target_parameters_reduce", default=0.0, type=float)
    parser.add_argument("--target_flops_reduce", default=0.0, type=float)
    parser.add_argument("--lambda_value", default=0.1, type=float)


def main(args, ini_time):
    global best_prec1
    best_prec1 = 0
    args.gpu = 0
    
    if args.seed is not None:
        seed = args.seed
        print('current seed', seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # TODO: Do we need deterministic in cudnn ? Double check
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        #os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        print("Seeded everything")

    start_epoch = 0
    model_state = None
    optimizer_state = None

    loss = nn.CrossEntropyLoss
    
    memory_format = (torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format)

    model_and_loss = ModelAndLoss(
        (args.arch, args.model_config, args.num_classes),
        loss,
        cuda=True,
        mask=True,
        data_path = args.data,
    )
    print("Model information:", args.arch, args.model_config, args.num_classes)

    # Create data loaders and optimizers as needed
    '''
    if args.data_backend == "pytorch":
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    
    '''
    train_loader, val_loader = dataloader.get_tiny_imagenet_loaders(1, 1, args.batch_size) 
 
    logger = log.Logger(args.print_freq, [], start_epoch=start_epoch - 1)
    logger.log_parameter(args.__dict__, verbosity=dllogger.Verbosity.DEFAULT)

    named_parameters = list(model_and_loss.model.named_parameters())
    trainable_parameters = [(n, v) for n, v in named_parameters if v.requires_grad]
     
    optimizer = get_optimizer(
        trainable_parameters,
        args.fp16,
        args.lr,
        args.momentum,
        args.weight_decay,
        nesterov=args.nesterov,
        bn_weight_decay=args.bn_weight_decay,
        state=optimizer_state,
        static_loss_scale=args.static_loss_scale,
        dynamic_loss_scale=args.dynamic_loss_scale,
    )
    
    #optimizer = torch.optim.SGD(model_and_loss.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    if args.lr_schedule == "step":
        assert args.lr_decay_epochs != ''
        lr_decay_epochs = [int(e) for e in list(args.lr_decay_epochs.split(","))]
        lr_policy = lr_step_policy(args.lr, lr_decay_epochs, args.lr_decay_rate, args.warmup, logger=logger)
        print("Learning rate decay epochs:", type(lr_decay_epochs), lr_decay_epochs)
    elif args.lr_schedule == "cosine":
        lr_policy = lr_cosine_policy(args.lr, args.warmup, args.epochs, logger=logger)
    elif args.lr_schedule == "linear":
        lr_policy = lr_linear_policy(args.lr, args.warmup, args.epochs, logger=logger)
    else:
        raise EOFError("args.lr_schedule is wrong!")       
        
    #model_and_loss.load_model_state(model_state)
    
    num_trainable_parameters = sum(p.numel() for p in model_and_loss.model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_parameters}")
    
    start_pruningRound = 0
    rewind_filename = f"rewindEpoch_{args.rewind_epoch}_round_0_checkpoint.pth.tar"
    num_pruningRound = args.num_pruningRound
    resample = True if "resample" in args.prune_type else False
    reinit = True if "reinit" in args.prune_type else False


    conv_threshold = 0.0 
    lambda_value = args.lambda_value
    para_remained_per_list = []
    metric_list = []
    acc_list = []
    conv_threshold_list = []
    lambda_list = []
    flops_remained_list = []
    minus_threshold_flag = False
    target_acc_loss = args.target_accloss * 100 if "givenAccLoss" in args.prune_type or "givenAccLossParaReduce" in args.prune_type else None 
    target_parameters_reduce = args.target_parameters_reduce * 100 if "givenParaReduce" in args.prune_type  or "givenAccLossParaReduce" in args.prune_type else None
    target_flops_reduce = args.target_flops_reduce * 100 if "givenFlopsReduce" in args.prune_type else None
    fix_conv_threshold = args.conv_threshold if "fixThreshold" in args.prune_type else None
    track_rounds_list = []
    count_reset_time = 0
    print(f"fix_conv_threshold: {fix_conv_threshold}, lambda_value: {lambda_value}; target_acc_loss: {target_acc_loss}; target_parameters_reduce: {target_parameters_reduce}; target_flops_reduce: {target_flops_reduce}")

    for curr_round in range(start_pruningRound, num_pruningRound):
        print("\n\ncurr_round: {}".format(curr_round))
        #print('current threshold: {}'.format(conv_threshold))
        if not curr_round == 0:
            #prune
            metric_list = pruning_utils.prune(
                curr_round,
                prune_type=args.prune_type,
                model=model_and_loss,
                resample=resample,
                reinit=reinit,
                args=args,
                conv_threshold=conv_threshold,
                power_value=args.power_value,
                )

            #Rewind
            trainable_parameters = [(n, v) for n, v in list(model_and_loss.get_model().named_parameters())  if v.requires_grad]
            #optimizer = torch.optim.SGD(model_and_loss.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
            
            optimizer = get_optimizer(
                trainable_parameters,
                args.fp16,
                args.lr,
                args.momentum,
                args.weight_decay,
                nesterov=args.nesterov,
                bn_weight_decay=args.bn_weight_decay,
                state=None,
                static_loss_scale=args.static_loss_scale,
                dynamic_loss_scale=args.dynamic_loss_scale,
            )
            
        TRIAL_NUM = 0
        print(f"--- Pruning Level [{TRIAL_NUM}:{curr_round}/{num_pruningRound}]: {1.0 * (0.8 ** (curr_round)) * 100:.1f}---")
        
        # calcualte number of parameters
        remained_per, remained_fc, remained_conv, compress_ratio = pruning_utils.print_nonzeros(model_and_loss.get_model(), curr_round, args)

        # calcualte flops
        flops_speedup = 0
        flops_remain = 0

        start_epoch = start_epoch if curr_round == 0 else args.rewind_epoch + 1

        end_epoch = args.epochs

        checkpoint_filename = f"pruningRound_{curr_round}_"+args.checkpoint_filename
        accuracy = train_loop(
            model_and_loss,
            optimizer,
            lr_policy,
            train_loader,
            val_loader,
            args.fp16,
            logger,
            should_backup_checkpoint(args),
            use_amp=args.amp,
            batch_size_multiplier=1,
            start_epoch=0,
            end_epoch=1,
            best_prec1=best_prec1,
            prof=args.prof,
            skip_training= True if curr_round == 0 else False,
            skip_validation=False,
            save_checkpoints=args.save_checkpoints and not args.evaluate,
            checkpoint_dir=args.workspace,
            checkpoint_filename=checkpoint_filename,
            rewind_filename=rewind_filename if curr_round == 0 else None,
            rewind_epoch=args.rewind_epoch,
            log_filename=args.workspace + f"/log/pruningRound_{curr_round}.log",
            remained_per=remained_per,
            compress_ratio=compress_ratio, 
            flops_speedup=flops_speedup, 
            flops_remain=flops_remain, 
            args = args,
            curr_round=curr_round,
        )

        #exit()
        # Calcualte global threshold and save it
        if "AAP" in args.prune_type:
            
            acc_list.append(accuracy)
            para_remained_per_list.append(remained_per)
            conv_threshold_list.append(conv_threshold)
            lambda_list.append(lambda_value)
            flops_remained_list.append(flops_remain)
            
            # Given target_acc_loss, calculate threshold
            if "_givenAccLoss_" in args.prune_type:
                target = target_acc_loss
                curr_reduction = acc_list[0] - acc_list[-1]

            elif "_givenParaReduce" in args.prune_type:
                target = target_parameters_reduce
                curr_reduction = para_remained_per_list[0] - para_remained_per_list[-1]

            elif "_givenFlopsReduce" in args.prune_type:
                target = target_flops_reduce
                curr_reduction = flops_remained_list[0] - para_remained_per_list[-1]

            elif "_givenAccLossParaReduce" in args.prune_type:
                target = target_acc_loss
                curr_reduction = acc_list[0] - acc_list[-1]
                one_condition = False
                target2 = target_parameters_reduce
                curr_reduction2 = para_remained_per_list[0] - para_remained_per_list[-1]

            else:
                raise EOFError(f"{args.prune_type} is not implemented!")
            
            lambda_value, conv_threshold, model_and_loss, track_rounds_list, count_reset_time = pruning_utils.apdaptive_pruning_given_target(
                args = args, 
                acc_list = acc_list,
                para_remained_per_list = para_remained_per_list,
                flops_remained_list = flops_remained_list,
                conv_threshold_list = conv_threshold_list,
                lambda_list = lambda_list,
                file_thresholds_content = None,
                curr_round = curr_round,
                model_and_loss = model_and_loss,
                track_rounds_list = track_rounds_list,
                count_reset_time = count_reset_time,
                target = target, # difference among three methods
                curr_reduction = curr_reduction, # difference among three methods
                one_condition = one_condition, # difference among one and two conditions
                target2 = target2, # difference among one and two conditions
                curr_reduction2 = curr_reduction2, # difference among one and two conditions
            )
 
            # do inference for 1 epoch
            _ = train_loop(
                model_and_loss,
                optimizer,
                lr_policy,
                train_loader,
                val_loader,
                args.fp16,
                None,
                should_backup_checkpoint(args),
                use_amp=args.amp,
                batch_size_multiplier=None,
                start_epoch=0,
                end_epoch=1,
                best_prec1=None,
                prof=args.prof,
                skip_training=True,
                skip_validation=False,
                save_checkpoints=None,
                checkpoint_dir=None,
                checkpoint_filename=None,
                rewind_filename=None,
                rewind_epoch=None,
                log_filename=None,
                remained_per=None,
                compress_ratio=None, 
                flops_speedup=None, 
                flops_remain=None,
                args = None,
                curr_round=None, 
            )

        round_time = time.time() - ini_time
        print("Util Current Round duration: {}s".format(round_time))

if __name__ == "__main__": 
    exp_start_time = time.time()
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    add_parser_arguments(parser)
    args = parser.parse_args()
    
    if args.target_parameters_reduce == 0.6:
        args.num_pruningRound = 30
        args.target_accloss = 0.1
    elif args.target_parameters_reduce == 0.7:
        args.num_pruningRound = 40
        args.target_accloss = 0.11
    elif args.target_parameters_reduce == 0.8:
        args.num_pruningRound = 90
        args.target_accloss = 0.15

    data_name = args.data.split("/")[-1]
    args.workspace = "./non_change_dense/logs/" 
    
    if not os.path.exists(args.workspace+"/log"):
        os.makedirs(args.workspace+"/log")
    #print("workspace: {}".format(args.workspace))

    main(args, exp_start_time)

    exp_duration = time.time() - exp_start_time
    print("Experiment ended, exp_duration: {}s".format(exp_duration))

