import os
import time
import utils
import torch
import datetime
from get_args_parser import get_args_parser
from load_data import load_data
from create_model import create_model
from get_optim import get_optim
from train_one_epoch import train_one_epoch
from evaluate import evaluate
from collections import OrderedDict
from transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import timm
import torch.nn as nn

torch.set_float32_matmul_precision('high')

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    data_loader, data_loader_test, train_sampler, test_sampler, num_classes = load_data(args)

    print("Creating model")
    model, policy_net = create_model(args, device)
    criterion, optimizer, pol_optimizer, main_lr_scheduler, pol_lr_scheduler = get_optim(model, policy_net, args)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        policy_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(policy_net)
        
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        policy_net = torch.nn.parallel.DistributedDataParallel(policy_net, device_ids=[args.gpu])
        policy_net_without_ddp = policy_net.module        

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])   
        policy_net_without_ddp.load_state_dict(checkpoint["policy_net"])
        base_alpha = checkpoint["base_alpha"]
        base_beta = checkpoint["base_beta"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        pol_optimizer.load_state_dict(checkpoint["pol_optimizer"])
        main_lr_scheduler.load_state_dict(checkpoint["main_lr_scheduler"])
        pol_lr_scheduler.load_state_dict(checkpoint["pol_lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate(model, criterion, data_loader_test, device=device)
        return
    
    base_alpha = torch.rand((1, args.grid * args.grid))
    base_beta = torch.rand((1, args.grid * args.grid))    

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_alpha, base_beta = train_one_epoch(model, policy_net, base_alpha, base_beta, criterion, optimizer, pol_optimizer, data_loader, device, epoch, args)
        main_lr_scheduler.step()
        pol_lr_scheduler.step()
        evaluate(model, policy_net, base_alpha, base_beta, criterion, data_loader_test, args, device)
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "policy_net": policy_net_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "pol_optimizer": pol_optimizer.state_dict(),
                "main_lr_scheduler": main_lr_scheduler.state_dict(),
                "pol_lr_scheduler": pol_lr_scheduler.state_dict(),
                "base_alpha": base_alpha,
                "base_beta": base_beta,
                "epoch": epoch,
                "args": args,
            }
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f'checkpoint_{args.model}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")    
    
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
