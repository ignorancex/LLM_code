import argparse
import copy
import datetime
from pathlib import Path
import time

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils.dataclasses import ProjectConfiguration

from util.crop import center_crop_arr
from util.loader import CachedFolder
from util.misc import add_weight_decay

from models.vae import AutoencoderKL
from models import create_mar
from engine_mar import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAR training with Diffusion Loss', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=400, type=int)

    # Model parameters
    parser.add_argument('--model', default='mar_large', type=str, metavar='MODEL',
                        help='Name of model to train')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='images input size')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    parser.add_argument('--vae_stride', default=16, type=int,
                        help='tokenizer stride, default use KL16')
    parser.add_argument('--patch_size', default=1, type=int,
                        help='number of tokens to group as a patch.')

    # Generation parameters
    parser.add_argument('--num_iter', default=64, type=int,
                        help='number of autoregressive iterations to generate an image')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='number of images to generate')
    parser.add_argument('--cfg', default=1.0, type=float, help="classifier-free guidance")
    parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--eval_freq', type=int, default=40, help='evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=5, help='save last frequency')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_bsz', type=int, default=64, help='generation batch size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')

    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--grad_accu', type=int, default=1,
                        help='gradient accumulation')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--ema_rate', default=0.9999, type=float)

    # MAR params
    parser.add_argument('--mask_ratio_min', type=float, default=0.7,
                        help='Minimum mask ratio')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='projection dropout')
    parser.add_argument('--buffer_size', type=int, default=64)

    # Diffusion Loss params
    parser.add_argument('--diffloss_d', type=int, default=12)
    parser.add_argument('--diffloss_w', type=int, default=1536)
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--diffusion_batch_mul', type=int, default=1)
    parser.add_argument('--temperature', default=1.0, type=float, help='diffusion loss sampling temperature')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--class_num', default=1000, type=int)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--pretrained', default='', help='pretrain mar model')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # caching latents
    parser.add_argument('--use_cached', action='store_true', dest='use_cached',
                        help='Use cached latents')
    parser.set_defaults(use_cached=False)
    parser.add_argument('--cached_path', default='', help='path to cached latents')

    return parser


def main(args):
    accelerator = Accelerator(log_with='tensorboard',
                              project_config=ProjectConfiguration(project_dir=args.output_dir,
                                                                  logging_dir=args.log_dir,
                                                                  total_limit=2,
                                                                  automatic_checkpoint_naming=True))
    accelerator.init_trackers(project_name=args.model + "_full")
    import builtins
    builtins.print = accelerator.on_main_process(print)

    accelerator.print(f'job dir: {Path("./").cwd()}')
    accelerator.print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    set_seed(args.seed)

    torch.backends.cudnn.benchmark = True

    # augmentation following DiT and ADM
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if args.use_cached:
        dataset_train = CachedFolder(args.cached_path)
    else:
        # dataset_train = datasets.ImageFolder(str(Path(args.data_path) / 'train'), transform=transform_train)
        dataset_train = range(10)
    accelerator.print(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=True,
        drop_last=True,
    )

    # define the vae and mar model
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).eval()
    for param in vae.parameters():
        param.requires_grad = False

    model = create_mar(args.model,
                       img_size=args.img_size,
                       vae_stride=args.vae_stride,
                       patch_size=args.patch_size,
                       vae_embed_dim=args.vae_embed_dim,
                       mask_ratio_min=args.mask_ratio_min,
                       label_drop_prob=args.label_drop_prob,
                       class_num=args.class_num,
                       attn_dropout=args.attn_dropout,
                       proj_dropout=args.proj_dropout,
                       buffer_size=args.buffer_size,
                       # diffloss_d=args.diffloss_d,
                       # diffloss_w=args.diffloss_w,
                       num_sampling_steps=args.num_sampling_steps,
                       diffusion_batch_mul=args.diffusion_batch_mul,
                       grad_checkpointing=args.grad_checkpointing,
                       )
    if args.pretrained:
        accelerator.print(f"Loading pretrained model from {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu')["model_ema"])

    accelerator.print(f"Model = {model}")
    # following timm: set wd as 0 for bias and norm layers
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Number of trainable parameters: {n_params / 1e6}M")

    eff_batch_size = args.batch_size * accelerator.num_processes

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    accelerator.print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    accelerator.print(f"actual lr: {args.lr:.2e}")
    accelerator.print(f"effective batch size: {eff_batch_size}")

    # no weight decay on bias, norm layers, and diffloss MLP
    param_groups = add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    accelerator.print(optimizer)
    scheduler_factor = len(data_loader_train)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                              start_factor=0.001,
                                              total_iters=args.warmup_epochs * scheduler_factor),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                       eta_min=args.min_lr,
                                                       T_max=(args.epochs - args.warmup_epochs) * scheduler_factor)
        ],
        milestones=[args.warmup_epochs * scheduler_factor])

    ema_model = copy.deepcopy(model)

    model, vae, data_loader_train, optimizer, scheduler, ema_model = accelerator.prepare(model, vae, data_loader_train,
                                                                                         optimizer,
                                                                                         scheduler, ema_model)

    # resume training
    if args.resume:
        checkpoint = torch.load(Path(args.resume)/"checkpoint-last.pth", map_location='cpu')
        accelerator.unwrap_model(model).load_state_dict(checkpoint['model'])
        ema_state_dict = checkpoint['model_ema']
        accelerator.unwrap_model(ema_model).load_state_dict(ema_state_dict)
        accelerator.print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            # if 'scaler' in checkpoint:
            #     loss_scaler.load_state_dict(checkpoint['scaler'])
            accelerator.print("With optim & sched!")
        del checkpoint
    else:
        # model_params = list(model_without_ddp.parameters())
        # ema_params = copy.deepcopy(model_params)
        accelerator.print("Training from scratch")

    # evaluate FID and IS
    if args.evaluate:
        accelerator.free_memory()
        evaluate(accelerator.unwrap_model(model), accelerator.unwrap_model(vae), accelerator.unwrap_model(ema_model),
                 args, 0, batch_size=args.eval_bsz, accelerator=accelerator,
                 cfg=args.cfg, use_ema=True)
        return

    # training
    accelerator.print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(
            model, vae,
            ema_model,
            data_loader_train,
            optimizer, scheduler, epoch,
            accelerator=accelerator,
            args=args
        )

        # save checkpoint
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            accelerator.save_state(safe_serialization=False)

        # online evaluation
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            accelerator.free_memory()
            evaluate(accelerator.unwrap_model(model), accelerator.unwrap_model(vae),
                     accelerator.unwrap_model(ema_model), args, epoch, batch_size=args.eval_bsz,
                     accelerator=accelerator,
                     cfg=1.0, use_ema=True)
            if not (args.cfg == 1.0 or args.cfg == 0.0):
                evaluate(accelerator.unwrap_model(model), accelerator.unwrap_model(vae),
                         accelerator.unwrap_model(ema_model), args, epoch, batch_size=args.eval_bsz // 2,
                         accelerator=accelerator, cfg=args.cfg, use_ema=True)
            accelerator.free_memory()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    accelerator.print(f'Training time {total_time_str}')
    accelerator.end_training()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # args.log_dir = args.output_dir
    main(args)
