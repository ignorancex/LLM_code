import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..')) # to get myutils

import argparse
import torch.optim as optim
import models
from pathlib import Path

from SynCDR.LinearAverage import LinearAverage
from return_dataset import return_dataset_selfsup
import logging
from SynCDR.utils.utils_dh import setup_logging

from train_fns.selfsup import train_selfsup_only

from retrieval import retrieval_recall
import copy

from myutils import ioutils
import shutil

import torch
torch.backends.cudnn.benchmark=True

def parse_args(args_str=None):
    # Training settings
    parser = argparse.ArgumentParser(description='Visda Classification')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                        help='learning rate multiplication')
    parser.add_argument('--T', type=float, default=0.05, metavar='T',
                        help='temperature (default: 0.05)')
    parser.add_argument('--T_syn', type=float, default=1., metavar='T',
                        help='temperature for real-syn matching (default: 1.)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--net', type=str, default='resnet50', metavar='B',
                        help='which network ')
    parser.add_argument('--source', type=str, default='Real', metavar='B',
                        help='board dir')
    parser.add_argument('--target', type=str, default='Clipart', metavar='B',
                        help='board dir')
    parser.add_argument('--dataset', type=str, default='office_home', choices=['office', 'office_home', 'cub', 'domainnet', 'adaptiope'],
                        help='the name of dataset, multi is large scale dataset')

    parser.add_argument('--low_dim', default=512, type=int,
                        metavar='D', help='feature dimension')
    parser.add_argument('--batch_size', default=16, type=int,
                        metavar='M', help='batch_size')
    parser.add_argument('--nce-t', default=0.1, type=float,
                        metavar='T', help='temperature parameter for softmax')
    parser.add_argument('--nce-m', default=0.5, type=float,
                        metavar='M', help='momentum for non-parametric updates')
    parser.add_argument('--n_neighbor', default=700, type=int,
                        metavar='M', help='Parameter of CDS. Max num neighbors to consider for entropy computation.')
    
    parser.add_argument('--resume', default='', type=str, help='Path to checkpoint to resume from')
    
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',)
    parser.add_argument('--max_epochs', type=int, default=15)
    
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Use wandb for logging')
    parser.add_argument('--wandb_project', default='CDS', type=str, help='Wandb project name')
    parser.add_argument('--expt_name', default='tmp', type=str, help='Experiment name')
    parser.add_argument('--save_dir', default='expts/tmp', type=str, help='Save directory for checkpoints')
    parser.add_argument('--log_freq', default=50, type=int, help='Log frequency for wandb')
    parser.add_argument('--eval_freq', default=1, type=int, help='Evaluation frequency')
    
    
    parser.add_argument('--train_split', default='train', type=str, help='Split to use for training (src domain)')
    parser.add_argument('--train_split2', default=None, type=str, help='Split to use for training (tgt domain)')
    parser.add_argument('--train_root', default=None, type=str, help='Root for training data if different from default="data"')
    parser.add_argument('--eval_split', default='val', type=str, help='Split to use for evaluation')
    parser.add_argument('--eval_root', default=None, type=str, help='Root for validation data if separate from training')
    parser.add_argument('--add_test_eval', action='store_true', default=False, help='Evaluate on test set as well')
    parser.add_argument('--test_split', default='test', type=str, help='Split to use for testing')
    
    parser.add_argument('--real_root', default='/home/ubuntu/data', type=str, help='Root directory of real data, explicitly provided for SyntheticDataset')
    parser.add_argument('--syn_root', default='../ELITE/synthetic_data', type=str, help='Root directory of synthetic data')
    parser.add_argument('--syn_wt', default=0.5, type=float, help='Weight for synthetic data loss')
    
    parser.add_argument('--ppp', action='store_true', default=False, help='Use PPP criterion')
    parser.add_argument('--no_cross', action='store_true', default=False, help='Do not use cross domain losses (does not include PPP)')

    args = parser.parse_args(args=args_str)
    
    return args

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    wandb = ioutils.WandbWrapper(debug=(not args.use_wandb))
    wandb.init(
        name=args.expt_name, project=args.wandb_project, 
        dir=args.save_dir, config=args, reinit=True)

    if args.syn_root:
        assert os.path.exists(args.syn_root), f'Synthetic data root {args.syn_root} does not exist'
        args.s2t = args.source[0] + '2' + args.target[0]
        args.t2s = args.target[0] + '2' + args.source[0]
        s2t_path = os.path.exists(os.path.join(args.syn_root, args.dataset, args.s2t))
        t2s_path = os.path.exists(os.path.join(args.syn_root, args.dataset, args.t2s))
        assert os.path.exists(s2t_path), f'Synthetic data path {s2t_path} does not exist'
        assert os.path.exists(t2s_path), f'Synthetic data path {t2s_path} does not exist'
        
    setup_logging(args.save_dir)
    
    if args.net == 'resnet50':
        net = models.__dict__['resnet50'](pretrained=True, low_dim=args.low_dim)
        preprocess_tr = None
        preprocess_val = None
    elif args.net.startswith('clip'):
        import open_clip
        if args.net == 'clip_resnet50':
            net, preprocess_tr, preprocess_val = open_clip.create_model_and_transforms('RN50', pretrained='openai')
            net = net.visual
            args.low_dim = open_clip.get_model_config('RN50')['embed_dim']
        elif args.net == 'clip_resnet101':
            net, preprocess_tr, preprocess_val = open_clip.create_model_and_transforms('RN101', pretrained='openai')
            net = net.visual
            args.low_dim = open_clip.get_model_config('RN101')['embed_dim']
        elif args.net == 'clip_vitb_laion2b':
            net, preprocess_tr, preprocess_val = open_clip.create_model_and_transforms('ViT-B/32', pretrained='laion2b_s34b_b79k')
            net = net.visual
            args.low_dim = open_clip.get_model_config('ViT-B-32')['embed_dim']
        elif args.net == 'clip_eva_vitg':
            net, preprocess_tr, preprocess_val = open_clip.create_model_and_transforms('EVA01-g-14', pretrained='laion400m_s11b_b41k')
            net = net.visual
            args.low_dim = open_clip.get_model_config('EVA01-g-14')['embed_dim']
        elif args.net == 'clip_eva_vitg_plus':
            net, preprocess_tr, preprocess_val = open_clip.create_model_and_transforms('EVA01-g-14-plus', pretrained='merged2b_s11b_b114k')
            net = net.visual
            args.low_dim = open_clip.get_model_config('EVA01-g-14-plus')['embed_dim']
    elif args.net.startswith('imagenet'):
        from timm import create_model
        if args.net == 'imagenet_vitb':
            net = create_model('vit_base_patch32_224.sam_in1k', pretrained=True, num_classes=0)
            args.low_dim = net.embed_dim
        elif args.net == 'imagenet21k_vitb':
            net = create_model('vit_base_patch32_224.augreg_in21k', pretrained=True, num_classes=0)
            args.low_dim = net.embed_dim
        elif args.net == 'imagenet_vits':
            net = create_model('vit_small_patch16_224.augreg_in1k', pretrained=True, num_classes=0)
            args.low_dim = net.embed_dim
        elif args.net == 'imagenet21k_vits':
            net = create_model('vit_small_patch16_224.augreg_in21k', pretrained=True, num_classes=0)
            args.low_dim = net.embed_dim
        elif args.net == 'imagenet_resnet101':
            net = create_model('resnet101.tv2_in1k', pretrained=True, num_classes=0)
            args.low_dim = net.num_features
        elif args.net == 'imagenet_resnet152':
            net = create_model('resnet152.tv2_in1k', pretrained=True, num_classes=0)
            args.low_dim = net.num_features
        preprocess_tr = None
        preprocess_val = None
    else:
        raise ValueError('Network cannot be recognized.')
    
    
    logging.info('Initializing datasets and loaders...')
    all_loaders = return_dataset_selfsup(
        args, batch_size=args.batch_size, train_split=args.train_split, 
        val_split=args.eval_split, train_root=args.train_root, val_root=args.eval_root,
        train_split2=args.train_split2, preprocess_tr=preprocess_tr, preprocess_val=preprocess_val)
    source_loader = all_loaders[0]
    target_loader = all_loaders[1]
    val_loader = all_loaders[2]
    source_loader_val = all_loaders[3]
    class_list = all_loaders[4]
    logging.info('Done')

    if args.syn_root:
        source_syn_loader = all_loaders[5]
        target_syn_loader = all_loaders[6]
    else:
        source_syn_loader = None
        target_syn_loader = None

    logging.info(' '.join(sys.argv))
    logging.info('dataset %s source %s target %s' % (args.dataset, args.source, args.target))
    torch.cuda.manual_seed(args.seed)
        
    net.cuda()
    net.train()    
    
    lemniscate_s = LinearAverage(args.low_dim, len(source_loader.dataset) , args.nce_t, args.nce_m)
    lemniscate_t = LinearAverage(args.low_dim, len(target_loader.dataset), args.nce_t, args.nce_m)
    lemniscate_s.cuda()
    lemniscate_t.cuda()

    
    logging.info('source data len : {}'.format(len(source_loader.dataset)))
    logging.info('target data len : {}'.format(len(target_loader.dataset)))

    all_trainable_params = list(net.parameters())
    optimizer = optim.SGD(all_trainable_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    best_metric = 0
    best_epoch = 0
    
    if args.resume:
        if not os.path.exists(args.resume):
            raise ValueError(f'Checkpoint {args.resume} does not exist')
        logging.info(f'Resuming from checkpoint {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        start_epoch = ckpt['epoch'] + 1
        
        net.load_state_dict(ckpt['net'])
        net.cuda()
        net.train()
        
        optimizer.load_state_dict(ckpt['optimizer'])


        lemniscate_s.load_state_dict(ckpt['lemniscate_s'])
        lemniscate_t.load_state_dict(ckpt['lemniscate_t'])
        lemniscate_s.cuda()
        lemniscate_t.cuda()
    else:
        start_epoch = 0
    
    
    for epoch in range(start_epoch, args.max_epochs):
        logging.info('epoch: start:{}'.format(epoch))
        log_info = train_selfsup_only(
            epoch, args, wandb, net, 
            lemniscate_s, lemniscate_t, optimizer, 
            source_loader, target_loader,
            source_syn_loader=source_syn_loader, target_syn_loader=target_syn_loader)
            
        log_info['Epoch'] = epoch
        # Print training part of the log info
        print(ioutils.get_log_str(log_info))
        
        if (epoch + 1) % args.eval_freq == 0:
            logging.info(f'Retrieval eval {args.source} -> {args.target}')
            curr_prec_s2t = retrieval_recall(epoch, args, net, all_loaders=all_loaders)
            log_info.update({'S2T ' + key : val for key, val in curr_prec_s2t.items()})
            if args.add_test_eval:
                curr_prec_s2t_test = retrieval_recall(
                    epoch, args, net, all_loaders=None, train_split=args.train_split, 
                    val_split=args.test_split, train_root=args.train_root, val_root=args.eval_root)
                log_info.update({'S2T Test ' + key : val for key, val in curr_prec_s2t_test.items()})

            rev_args = copy.deepcopy(args)
            rev_args.source = args.target
            rev_args.target = args.source
            rev_args.s2t = rev_args.source[0] + '2' + rev_args.target[0]
            rev_args.t2s = rev_args.target[0] + '2' + rev_args.source[0]
            all_loaders1 = [all_loaders[1], all_loaders[0], all_loaders[3], all_loaders[2], all_loaders[4]]
            if args.syn_root:
                all_loaders1.extend([all_loaders[6], all_loaders[5]])
            logging.info(f'Retrieval eval {rev_args.source} -> {rev_args.target}')
            curr_prec_t2s = retrieval_recall(epoch, rev_args, net, all_loaders=tuple(all_loaders1))
            if args.add_test_eval:
                curr_prec_t2s_test = retrieval_recall(
                    epoch, rev_args, net, all_loaders=None, train_split=args.train_split, 
                    val_split=args.test_split, train_root=args.train_root, val_root=args.eval_root)
                log_info.update({'T2S Test ' + key : val for key, val in curr_prec_t2s_test.items()})
            
            log_info.update({'T2S ' + key : val for key, val in curr_prec_t2s.items()})
            
            curr_metric = (next(iter(curr_prec_s2t.values())) + next(iter(curr_prec_t2s.values()))) / 2.
            log_info.update({'Metric' : curr_metric})
            if args.add_test_eval:
                curr_metric_test = (next(iter(curr_prec_s2t_test.values())) + next(iter(curr_prec_t2s_test.values()))) / 2.
                log_info.update({'Metric Test' : curr_metric_test})
            wandb.log(log_info)
        
            # save model state
            save_dict = {
                'epoch' : epoch,
                'net' : net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'curr_metric' : curr_metric,
                'best_metric' : best_metric,
                'best_epoch' : best_epoch,
            }
            save_dict.update({
                'lemniscate_s' : lemniscate_s.state_dict(),
                'lemniscate_t' : lemniscate_t.state_dict(),
            })
            torch.save(save_dict, os.path.join(args.save_dir, 'checkpoint.pth.tar'))
            
            # copy to best model
            if curr_metric > best_metric:
                best_metric = curr_metric
                best_epoch = epoch
                shutil.copyfile(
                    os.path.join(args.save_dir, 'checkpoint.pth.tar'),
                    os.path.join(args.save_dir, 'best_model.pth.tar'))
            
    wandb.join()

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
        

