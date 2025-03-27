from hcompnet.model import HComPNet, get_network
from util.log import Log
import torch.nn as nn
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders, SubsetSequentialSampler, unshuffle_dataloader, create_filtered_dataloader
from util.func import init_weights_xavier
from hcompnet.train_and_test import train, test
import torch
import sys, os
import random
import numpy as np
from shutil import copy
import matplotlib.pyplot as plt
from copy import deepcopy

from omegaconf import OmegaConf
from util.node import Node
import shutil
from util.phylo_utils import construct_phylo_tree, construct_discretized_phylo_tree
from util.custom_losses import WeightedCrossEntropyLoss, WeightedNLLLoss, FocalLossWrapper
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, SubsetRandomSampler

import time
import wandb
from collections import Counter


def run_pipnet(args=None):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args = args or get_args()
    assert args.batch_size > 1

    if args.wandb == 'n':
        os.environ['WANDB_DISABLED'] = 'true'
        print('Disabled wandb')
 
    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    save_args(args, log.metadata_dir)

    wandb_run = wandb.init(project="pipnet", name=os.path.basename(args.log_dir), config=vars(args), reinit=False)

    phylo_config = OmegaConf.load(args.phylo_config)
    # construct the phylo tree
    if phylo_config.phyloDistances_string == 'None':
        root = construct_phylo_tree(phylo_config.phylogeny_path)
        print('-'*25 + ' No discretization ' + '-'*25)
    else:
        root = construct_discretized_phylo_tree(phylo_config.phylogeny_path, phylo_config.phyloDistances_string)
        print('-'*25 + ' Discretized ' + '-'*25)
    root.assign_all_descendents()

    # update num of protos per node based on num_protos_per_descendant
    for node in root.nodes_with_children():
        node.set_num_protos(num_protos_per_descendant=args.num_protos_per_descendant,\
                            num_protos_per_child=args.num_protos_per_child,\
                            min_protos_per_child=args.min_protos_per_child)

    gpu_list = args.gpu_ids.split(',')
    device_ids = []
    if args.gpu_ids!='':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))
    
    global device
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids)==1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
        elif len(device_ids)==0:
            device = torch.device('cuda')
            print("CUDA device set without id specification", flush=True)
            device_ids.append(torch.cuda.current_device())
        else:
            device_str = ''
            for d in device_ids:
                device_str+=str(d)
                device_str+=","
            device = torch.device('cuda:'+str(device_ids[0]))
    else:
        device = torch.device('cpu')
     
    # Log which device was actually used
    print("Device used: ", device, "with id", device_ids, flush=True)
    
    # Obtain the dataset and dataloaders
    trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device)
    print("Classes: ", str(classes), flush=True)

    if ('leave_out_classes' in args) and (args.leave_out_classes != ''):
        with open(args.leave_out_classes, 'r') as file:
            leave_out_classes = [line.strip() for line in file]

        leave_out_loader = testloader
        classes_to_keep = leave_out_classes
        idx_of_classes_to_keep = set()
        name2label = leave_out_loader.dataset.class_to_idx # param
        label2name = {label:name for name, label in name2label.items()}
        for label in label2name:
            # NOTE: Keeping the left out classes here
            if label2name[label] in classes_to_keep:
                idx_of_classes_to_keep.add(label)

        target_indices = []
        for i in range(len(leave_out_loader.dataset)):
            *_, label = leave_out_loader.dataset[i]
            if label in idx_of_classes_to_keep:
                target_indices.append(i)
        sampler = SubsetRandomSampler(target_indices)
        to_shuffle = False

        leave_out_loader = create_filtered_dataloader(leave_out_loader, sampler)

    print("Node count:", len(root.nodes_with_children()))

    if ('weighted_ce_loss' in args) and (args.weighted_ce_loss == 'y'):
        for node in root.nodes_with_children():
            node.set_loss_weightage_using_descendants_count()

    # Create a convolutional network based on arguments and add 1x1 conv layer
    feature_net, add_on_layers, pool_layer, classification_layers = get_network(args, root)

    net = HComPNet(feature_net = feature_net,
                    args = args,
                    add_on_layers = add_on_layers,
                    pool_layer = pool_layer,
                    classification_layers = classification_layers,
                    num_parent_nodes = len(root.nodes_with_children()),
                    root = root
                    )
    net = net.to(device=device)
    net = nn.DataParallel(net, device_ids = device_ids)    

    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)   

    # Initialize or load model
    with torch.no_grad():
        if args.state_dict_dir_net != '':
            raise Exception('Do not use this, use state_dict_dir_backbone for loading pretrained ._net')
            checkpoint = torch.load(args.state_dict_dir_net,map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'],strict=True) 
            print("Pretrained network loaded", flush=True)
            net.module._multiplier.requires_grad = False
            try:
                optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict']) 
            except:
                print('-'*25, 'Unable to load optimizer_net_state_dict')
                pass

            loading_pretrained_only_model = False
            for attr in dir(net.module):
                if attr.endswith('_classification'):
                    # assume that the linear classification layer is not yet trained (e.g. when loading a pretrained backbone only)
                    if torch.mean(getattr(net.module, attr).weight).item() > 1.0 \
                        and torch.mean(getattr(net.module, attr).weight).item() < 3.0 \
                            and torch.count_nonzero(torch.relu(getattr(net.module, attr).weight-1e-5)).float().item() > 0.8*(getattr(net.module, attr).weight.shape[0] * getattr(net.module, attr).weight.shape[1]): 
                        print(f"We assume that the {attr} layer is not yet trained. We re-initialize it...", flush=True)
                        torch.nn.init.normal_(getattr(net.module, attr).weight, mean=1.0,std=0.1) 
                        torch.nn.init.constant_(net.module._multiplier, val=2.)
                        print(f"{attr} layer initialized with mean", torch.mean(getattr(net.module, attr).weight).item(), flush=True)
                        if args.bias:
                            torch.nn.init.constant_(getattr(net.module, attr).bias, val=0.)
                        loading_pretrained_only_model = True
            if loading_pretrained_only_model and 'optimizer_classifier_state_dict' in checkpoint.keys():
                optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])

        elif args.state_dict_dir_backbone != '':
            checkpoint = torch.load(args.state_dict_dir_backbone,map_location=device)
            # load feature-extractor 'module._net' and prototype vectors
            filtered_checkpoint_dict = {key:val for key, val in checkpoint['model_state_dict'].items() if (key.startswith('module._net') or key.startswith('module._add_on'))}
            net.load_state_dict(filtered_checkpoint_dict,strict=False) 
            print(f"Feature-extractor and prototype vectors loaded from {args.state_dict_dir_backbone}", flush=True)
            
            # initialize multiplier
            torch.nn.init.constant_(net.module._multiplier, val=2.)
            net.module._multiplier.requires_grad = False

        else:
            # initialize add on
            for attr in dir(net.module):
                if attr.endswith('_add_on'):
                    getattr(net.module, attr).apply(init_weights_xavier)

            # initialize multiplier
            torch.nn.init.constant_(net.module._multiplier, val=2.)
            net.module._multiplier.requires_grad = False


    # Define classification loss function and scheduler
    # if args.weighted_ce_loss == 'n' input weights during forward call will be none and the output will be unweighted mean
    criterion = WeightedNLLLoss(device=device).to(device)
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader_pretraining)*args.epochs_pretrain, eta_min=args.lr_block/100., last_epoch=-1)

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        xs1, _, _ = next(iter(trainloader))
        xs1 = xs1.to(device)
        features, proto_features, _, _ = net(xs1, inference=True)
        wshape = proto_features['root'].shape[-1]
        args.wshape = wshape #needed for calculating image patch size
        print("Output shape: ", proto_features['root'].shape, flush=True)
    
    # ------------------------- PRETRAINING PROTOTYPES PHASE -------------------------
    for epoch in range(1, args.epochs_pretrain+1):
        for param in params_to_train:
            param.requires_grad = True
        for attr in dir(net.module):
            if attr.endswith('_add_on'):
                for param in getattr(net.module, attr).parameters():
                    param.requires_grad = True
        for attr in dir(net.module):
            if attr.endswith('_classification'):
                for param in getattr(net.module, attr).parameters():
                    param.requires_grad = False
        for param in params_to_freeze:
            param.requires_grad = True # can be set to False when you want to freeze more layers
        for param in params_backbone:
            param.requires_grad = False #can be set to True when you want to train whole backbone (e.g. if dataset is very different from ImageNet)
        
        print("\nPretrain Epoch", epoch, "with batch size", trainloader_pretraining.batch_size, flush=True)

        train_info = train(net, trainloader_pretraining, optimizer_net, optimizer_classifier, scheduler_net, None, criterion, \
                            epoch, device, pretrain=True, finetune=False, wandb_logging=False, \
                            wandb_run=None, pretrain_epochs=args.epochs_pretrain, args=args)

    
    if args.state_dict_dir_net == '':
        net.eval()
        torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_pretrained'))
        net.train()

    # ------------------------- SECOND TRAINING PHASE -------------------------
    # re-initialize optimizers and schedulers for second training phase
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)            
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader)*args.epochs, eta_min=args.lr_net/100.)
    # scheduler for the classification layer is with restarts, such that the model can re-active zeroed-out prototypes. Hence an intuitive choice. 
    if args.epochs<=30:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=5, eta_min=0.001, T_mult=1, verbose=False)
    else:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=10, eta_min=0.001, T_mult=1, verbose=False)
    
    frozen = True
    for epoch in range(1, args.epochs + 1):                      
        if epoch <= args.epochs_finetune_classifier:
            for attr in dir(net.module):
                if attr.endswith('_classification'):
                    for param in getattr(net.module, attr).parameters():
                        param.requires_grad = True
            for attr in dir(net.module):
                if attr.endswith('_add_on'):
                    for param in getattr(net.module, attr).parameters():
                        param.requires_grad = False # True # False
            for param in params_to_train:
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = False
            for param in params_backbone:
                param.requires_grad = False
            finetune = True
        elif epoch > args.epochs_finetune_mask:
            for attr in dir(net.module):
                if attr.endswith('_classification'):
                    for param in getattr(net.module, attr).parameters():
                        param.requires_grad = False
            for attr in dir(net.module):
                if attr.endswith('_add_on'):
                    for param in getattr(net.module, attr).parameters():
                        param.requires_grad = False # False
            for param in params_to_train:
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = False
            for param in params_backbone:
                param.requires_grad = False
            for attr in dir(net.module):
                if attr.endswith('_proto_presence'):
                    param = getattr(net.module, attr)
                    param.requires_grad = True
        else:
            finetune=False          
            if frozen:
                # unfreeze backbone
                if epoch>(args.freeze_epochs):
                    # for param in net.module._add_on.parameters():
                    #     param.requires_grad = True
                    for attr in dir(net.module):
                        if attr.endswith('_add_on'):
                            for param in getattr(net.module, attr).parameters():
                                param.requires_grad = True
                    for param in params_to_freeze:
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = True   
                    frozen = False
                # freeze first layers of backbone, train rest
                else:
                    for param in params_to_freeze:
                        param.requires_grad = True #Can be set to False if you want to train fewer layers of backbone
                    # for param in net.module._add_on.parameters():
                    #     param.requires_grad = True
                    for attr in dir(net.module):
                        if attr.endswith('_add_on'):
                            for param in getattr(net.module, attr).parameters():
                                param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = False
        
        print("\n Epoch", epoch, "frozen:", frozen, flush=True)            
        if (epoch==args.epochs or epoch%30==0) and args.epochs>1:
            # SET SMALL WEIGHTS TO ZERO
            with torch.no_grad():
                torch.set_printoptions(profile="full")
                for attr in dir(net.module):
                    if attr.endswith('_classification'):
                        getattr(net.module, attr).weight.copy_(torch.clamp(getattr(net.module, attr).weight.data - 0.001, min=0.)) 
                        print(f"{attr} weights: ", getattr(net.module, attr).weight[getattr(net.module, attr).weight.nonzero(as_tuple=True)], \
                              (getattr(net.module, attr).weight[getattr(net.module, attr).weight.nonzero(as_tuple=True)]).shape, flush=True)
                        if args.bias:
                            print(f"{attr} bias: ", getattr(net.module, attr).bias, flush=True)
                torch.set_printoptions(profile="default")

            for node in root.nodes_with_children():
                classification_weights = getattr(net.module, '_'+node.name+'_classification').weight
                node_label_to_children = {label: name for name, label in node.children_to_labels.items()}
                for class_label in range(classification_weights.shape[0]):
                    class_name = node_label_to_children[class_label]
                    print(f'Num protos for {node.name} class', class_name, torch.nonzero(classification_weights[class_label, :] > 1e-3).shape[0])

        train_info = train(net, trainloader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, \
                            epoch, device, pretrain=False, finetune=finetune, wandb_logging=False, \
                            wandb_run=None, pretrain_epochs=args.epochs_pretrain, args=args)
        

        if (epoch==args.epochs or epoch%5==0) and args.epochs>1:
            test_info = test(net, testloader, criterion, epoch, device, pretrain=False, finetune=finetune, wandb_logging=False, \
                                wandb_run=None, pretrain_epochs=args.epochs_pretrain, args=args)

        with torch.no_grad():
            net.eval()
            torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained'))
            if epoch%30 == 0:
                net.eval()
                torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_%s'%str(epoch)))            
        
    net.eval()
    torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_last'))

    print("Done!", flush=True)

class Tee(object):
    def __init__(self, name, mode, outstream):
        self.file = open(name, mode)
        self.stdout = outstream
    def __del__(self):
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()

if __name__ == '__main__':
    time_ = time.time()
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print('manual_seed', (time.time()-time_)/60)

    time_ = time.time()
    print_dir = os.path.join(args.log_dir,'out.txt')
    tqdm_dir = os.path.join(args.log_dir,'tqdm.txt')
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    
    # sys.stdout.close()
    # sys.stderr.close()
    # sys.stdout = open(print_dir, 'a')
    # sys.stderr = open(tqdm_dir, 'a')

    sys.stdout = Tee(print_dir, 'a', sys.stdout)
    sys.stderr = Tee(tqdm_dir, 'a', sys.stderr)
    print('stderr', (time.time()-time_)/60)

    time_ = time.time()
    run_pipnet(args)
    print('Finished in', int(time.time()-time_)//60, 'minutes')
    
    # sys.stdout.close()
    # sys.stderr.close()
