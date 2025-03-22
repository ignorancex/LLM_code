import os
import sys
sys.path.append(os.path.abspath('.')) # CDS_pretraining
sys.path.append(os.path.abspath('..')) # project root

try:
    from myutils.launch_utils import hold_gpu
    hold_gpu()
except:
    pass

import numpy as np
import torch
from return_dataset import return_dataset_selfsup
import torch.nn.functional as F
import argparse
import models
import logging
from SynCDR.utils.utils_dh import setup_logging
from SynCDR.LinearAverage import LinearAverage
from collections import OrderedDict
from myutils import ioutils
import copy

def retrieval_recall(
    epoch, args, net, all_loaders=None, train_split='train', 
    val_split='val', train_root=None, val_root=None, preprocess_val=None):

    from sklearn.metrics.pairwise import cosine_similarity
    from PIL import Image
    
    net.eval()
    
    if all_loaders is None:
        all_loaders = return_dataset_selfsup(
            args, batch_size=args.batch_size, train_split=train_split, 
            val_split=val_split, train_root=train_root, val_root=val_root,
            preprocess_val=preprocess_val)
        
    target_val_loader = all_loaders[2]
    source_val_loader = all_loaders[3]
    
    # Simply reusing the memory code for getting features
    lemniscate_s = LinearAverage(args.low_dim, len(source_val_loader.dataset), args.nce_t, args.nce_m)
    lemniscate_t = LinearAverage(args.low_dim, len(target_val_loader.dataset), args.nce_t, args.nce_m)

    with torch.no_grad():
        recompute_memory(net, lemniscate_s, source_val_loader)
        recompute_memory(net, lemniscate_t, target_val_loader)

    s_feat = lemniscate_s.memory.cpu().numpy()
    t_feat = lemniscate_t.memory.cpu().numpy()

    sims = cosine_similarity(s_feat, t_feat)

    ret_dict = OrderedDict()
    for k in [1, 5, 15]:

        total_num = 0
        positive_num = 0
        for index in range(0,len(sims)):

            i = sims[index]

            args_ = np.argsort(-i)
            val_true = source_val_loader.dataset.labels[index]

            temp_total = min(k, (target_val_loader.dataset.labels == val_true).sum())

            preds = target_val_loader.dataset.labels[args_[:temp_total]]

            total_num += temp_total
            try:
                positive_num += (preds == val_true).sum()
            except:
                print('except')

        curr_prec = positive_num / float(total_num)
        ret_dict[f'Prec@{k}'] = curr_prec
        logging.info('Epoch: {}, precision at {} : {:.4f}'.format(epoch, k, curr_prec))
        
    return ret_dict
        
def recompute_memory(net, lemniscate, trainloader):
    net.eval()
    trainFeatures = lemniscate.memory.t()
    batch_size = 100

    with torch.no_grad():
        transform_bak = trainloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):

            targets = targets.cuda()
            inputs = inputs.cuda()
            c_batch_size = inputs.size(0)
            features = net(inputs)
            features = F.normalize(features)

            trainFeatures[:, batch_idx * batch_size:batch_idx * batch_size + c_batch_size] = features.data.t()

        trainLabels = torch.LongTensor(temploader.dataset.labels).cuda()
        trainloader.dataset.transform = transform_bak

    lemniscate.memory = trainFeatures.t()

    lemniscate.memory_first = False
    
def parse_args(ret='parsed'):
    parser = argparse.ArgumentParser('Arguments to evaluate retrieval')
    parser.add_argument('--net', type=str, default='resnet50')
    parser.add_argument('--low_dim', type=int, default=512)
    parser.add_argument('--dataset', type=str, default='office_home')
    parser.add_argument('--source', type=str, default='Real')
    parser.add_argument('--target', type=str, default='Clipart')
    parser.add_argument('--ckpt_path', default='', type=str, help='path to checkpoint')
    parser.add_argument('--batch_size', default=16, type=int,metavar='M', help='batch_size')
    parser.add_argument('--pretrained', action='store_true', default=False, help='use pretrained model')
    
    parser.add_argument('--nce-t', default=0.1, type=float,
                        metavar='T', help='temperature parameter for softmax')
    parser.add_argument('--nce-m', default=0.5, type=float,
                        metavar='M', help='momentum for non-parametric updates')
    
    parser.add_argument('--syn_root', type=str, default=None, help='path to synthetic data')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',)
    
    # note :  following is unused but prevent error when domain_train.txt is not present
    parser.add_argument('--train_split', default='test', type=str, help='Split to use for training') 
    
    parser.add_argument('--eval_split', default='test', type=str, help='Split to use for evaluation')
    parser.add_argument('--eval_root', default=None, type=str, help='Root for validation data if separate from training')
    
    # Wandb options
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Use wandb for logging')
    parser.add_argument('--wandb_project', default='CDS', type=str, help='Wandb project name')
    parser.add_argument('--expt_name', default='tmp', type=str, help='Experiment name')
    parser.add_argument('--save_dir', default='expts/tmp', type=str, help='Save directory for checkpoints')
    
    if ret == 'parsed':
        return parser.parse_args()
    elif ret == 'default':
        return parser.parse_args([])
    elif ret == 'parser':
        return parser    

def main(args):
    try:
        import open_clip
    except:
        print('open_clip not found')
        
    os.makedirs(args.save_dir, exist_ok=True)
    wandb = ioutils.WandbWrapper(debug=(not args.use_wandb))
    wandb.init(
        name=args.expt_name, project=args.wandb_project, 
        dir=args.save_dir, config=args, reinit=True)
    setup_logging(args.save_dir)
    
    if args.net == 'resnet50':
        net = models.__dict__['resnet50'](
            pretrained=args.pretrained, low_dim=args.low_dim)
        val_transform = None
    elif args.net == 'resnet50_conv':
        net = models.__dict__['resnet50_conv'](
            pretrained=args.pretrained)
        args.low_dim = 2048
        val_transform = None
    elif args.net.startswith('clip'):
        if args.net == 'clip_resnet50':
            net, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
            net = net.visual
            args.low_dim = open_clip.get_model_config('RN50')['embed_dim']
        elif args.net == 'clip_resnet101':
            net, _, preprocess = open_clip.create_model_and_transforms('RN101', pretrained='openai')
            net = net.visual
            args.low_dim = open_clip.get_model_config('RN101')['embed_dim']
        elif args.net == 'clip_vitb_openai':
            net, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
            net = net.visual
            args.low_dim = open_clip.get_model_config('ViT-B-32')['embed_dim']
        elif args.net == 'clip_vitl_openai':
            net, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
            net = net.visual
            args.low_dim = open_clip.get_model_config('ViT-L-14')['embed_dim']
        elif args.net == 'clip_vitb_laion2b':
            net, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='laion2b_s34b_b79k')
            net = net.visual
            args.low_dim = open_clip.get_model_config('ViT-B-32')['embed_dim']
        elif args.net == 'clip_eva_vitg':
            net, _, preprocess = open_clip.create_model_and_transforms('EVA01-g-14', pretrained='laion400m_s11b_b41k')
            net = net.visual
            args.low_dim = open_clip.get_model_config('EVA01-g-14')['embed_dim']
        elif args.net == 'clip_eva_vitg_plus':
            net, _, preprocess = open_clip.create_model_and_transforms('EVA01-g-14-plus', pretrained='merged2b_s11b_b114k')
            net = net.visual
            args.low_dim = open_clip.get_model_config('EVA01-g-14-plus')['embed_dim']
        val_transform = preprocess
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
        else:
            raise Exception(f'Network {args.net} not supported')
        val_transform = None 
    else:
        raise Exception(f'Network {args.net} not supported')

    if args.ckpt_path:
        if not os.path.exists(args.ckpt_path):
            raise Exception(f'Checkpoint path ({args.ckpt_path}) does not exist')
        else:
            ckpt = torch.load(args.ckpt_path)
            net.load_state_dict(ckpt['net'])
    
    net.eval()
    net.cuda()
    
    log_info = OrderedDict({})
    s2t_precs = retrieval_recall(
        -1, args, net, train_split=args.train_split, 
        val_split=args.eval_split, val_root=args.eval_root,
        preprocess_val=val_transform)
    log_info.update({'S2T ' + key : val for key, val in s2t_precs.items()})
    
    rev_args = copy.deepcopy(args)
    rev_args.source = args.target
    rev_args.target = args.source
    rev_args.s2t = rev_args.source[0] + '2' + rev_args.target[0]
    rev_args.t2s = rev_args.target[0] + '2' + rev_args.source[0]
    
    t2s_precs = retrieval_recall(
        -1, rev_args, net, train_split=rev_args.train_split, 
        val_split=rev_args.eval_split, val_root=rev_args.eval_root,
        preprocess_val=val_transform)
    
    log_info.update({'T2S ' + key : val for key, val in t2s_precs.items()})
    wandb.log(log_info)
    wandb.join()
        
if __name__ == '__main__':
    args = parse_args()
    main(args)