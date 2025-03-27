import os
import argparse
import pickle
import numpy as np
import random
import torch
import torch.optim
# from torchlars import LARS

"""
    Utility functions for handling parsed arguments
"""

def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Train a HComP-Net')
    parser.add_argument('--dataset',
                        type=str,
                        default='CUB-190-imgnet-224',
                        help='Data set on HComP-Net should be trained')
    parser.add_argument('--validation_size',
                        type=float,
                        default=0.,
                        help='Split between training and validation set. Can be zero when there is a separate test or validation directory. Should be between 0 and 1. Used for partimagenet (e.g. 0.2)')
    parser.add_argument('--net',
                        type=str,
                        default='convnext_tiny_26',
                        help='Base network used as backbone of HComP-Net. Default is convnext_tiny_26 with adapted strides to output 26x26 latent representations. Other option is convnext_tiny_13 that outputs 13x13 (smaller and faster to train, less fine-grained)')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size when training the model using minibatch gradient descent. Batch size is multiplied with number of available GPUs')
    parser.add_argument('--batch_size_pretrain',
                        type=int,
                        default=128,
                        help='Batch size when pretraining the prototypes (first training stage)')
    parser.add_argument('--epochs',
                        type=int,
                        default=60,
                        help='The number of epochs HComP-Net should be trained (second training stage)')
    parser.add_argument('--epochs_pretrain',
                        type=int,
                        default = 10,
                        help='Number of epochs to pre-train the prototypes (first training stage). Recommended to train at least until the align loss < 1'
                        )
    parser.add_argument('--epochs_finetune_classifier',
                        type=int,
                        default=3,
                        help='The number of epochs to finetune only classifier')
    parser.add_argument('--epochs_finetune_mask',
                        type=int,
                        default=float('inf'),
                        help='Only the mask will be trained after this epoch')
    parser.add_argument('--freeze_epochs',
                        type=int,
                        default = 10,
                        help='Number of epochs where pretrained features_net will be frozen while training classification layer (and last layer(s) of backbone)'
                        )
    parser.add_argument('--optimizer',
                        type=str,
                        default='Adam',
                        help='The optimizer that should be used when training HComP-Net')
    parser.add_argument('--lr',
                        type=float,
                        default=0.05, 
                        help='The optimizer learning rate for training the weights from prototypes to classes')
    parser.add_argument('--lr_block',
                        type=float,
                        default=0.0005, 
                        help='The optimizer learning rate for training the last conv layers of the backbone')
    parser.add_argument('--lr_net',
                        type=float,
                        default=0.0005, 
                        help='The optimizer learning rate for the backbone. Usually similar as lr_block.') 
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='Weight decay used in the optimizer')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./runs/experiment',
                        help='The directory in which train progress should be logged')
    parser.add_argument('--min_protos_per_child',
                        type=int,
                        default = 0,
                        help='Number of prototypes. When zero (default) the number of prototypes is the number of output channels of backbone. If this value is set, then a 1x1 conv layer will be added. Recommended to keep 0, but can be increased when number of classes > num output channels in backbone.')
    parser.add_argument('--num_protos_per_descendant',
                        type=int,
                        default=0,
                        help='Used for deciding the num of protos to assign for each node based on the number of descendants.')
    parser.add_argument('--num_protos_per_child',
                        type=int,
                        default=0,
                        help='Used for deciding the num of protos to assign for each node based on the number of descendants.')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Input images will be resized to --image_size x --image_size (square). Code only tested with 224x224, so no guarantees that it works for different sizes.')
    parser.add_argument('--state_dict_dir_net',
                        type=str,
                        default='',
                        help='The directory containing a state dict with a pretrained HComP-Net. E.g., ./runs/experiment/checkpoints/net_pretrained')
    parser.add_argument("--state_dict_dir_backbone",
                        type=str,
                        default='', 
                        help='The directory containing a state dict with a pretrained HComP-Net. Only the backbone i.e. "_net" will be loaded')
    parser.add_argument('--dir_for_saving_images',
                        type=str,
                        default='visualization_results',
                        help='Directoy for saving the prototypes and explanations')
    parser.add_argument('--disable_pretrained',
                        action='store_true',
                        help='When set, the backbone network is initialized with random weights instead of being pretrained on another dataset).'
                        )
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Random seed. Note that there will still be differences between runs due to nondeterminism.')
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='',
                        help='ID of gpu. Can be separated with comma')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Num workers in dataloaders.')
    parser.add_argument("--phylo_config",
                        type=str,
                        default=None, 
                        help='path to the yaml file containing "phylogeny_path" and "phyloDistances_string"') # "./configs/cub27_phylogeny.yaml"
    parser.add_argument('--orth_weight',
                        type=float,
                        default=0.1,
                        help='Weight of prototype orthogonality loss'
                        )
    parser.add_argument('--ovsp_weight',
                        type=float,
                        default=0.05,
                        help='Weight for overspecificity loss.'
                        )
    parser.add_argument('--disc_weight',
                        type=float,
                        default=0.1,
                        help='(y/n) Flag that indicates whether to use minimize_contrasting_set (minimize max activation for contrasting set) loss or not.'
                        )
    parser.add_argument('--wandb',
                        type=str,
                        default='n',
                        help='(y/n) Flag to enable/disable wandb logging'
                        )
    parser.add_argument('--weighted_ce_loss',
                        type=str,
                        default='y',
                        help='(y/n) Flag to indicate whether to use weighted loss for classification. This actually uses weighted NLLLoss'
                        )
    parser.add_argument('--leave_out_classes',
                        type=str,
                        default='',
                        help='Comma seperated list of class names'
                        )
    parser.add_argument('--mask_prune_overspecific',
                        type=str,
                        default='n',
                        help='Whether to learn a mask for pruning overspecific prototypes'
                        )
    parser.add_argument('--cl_weight',
                        type=float,
                        default=2.0,
                        help='Weight for classification loss')
    parser.add_argument('--bias',
                        action='store_true',
                        help='Flag that indicates whether to include a trainable bias in the linear classification layer.'
                        )
    
    args = parser.parse_args()
    if len(args.log_dir.split('/'))>2:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)


    return args


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)


def is_bias_or_batchnorm(param_name):
    return 'bias' in param_name or 'BatchNorm' in param_name

# Create a parameter group excluding bias and batchnorm parameters for weight decay
def exclude_bias_and_batchnorm(named_params):
    params = []
    excluded_params = []
    for name, param in named_params.items():
        if not param.requires_grad:
            continue  # Skip frozen weights
        if is_bias_or_batchnorm(name):
            # Exclude from LARS adaptation and weight decay
            excluded_params.append(param)
        else:
            # Include in LARS adaptation and weight decay
            params.append(param)
    return params, excluded_params


def get_optimizer_nn(net, args: argparse.Namespace) -> torch.optim.Optimizer:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    #create parameter groups
    params_to_freeze = []
    params_to_train = []
    params_backbone = []
    # set up optimizer
    if 'convnext' in args.net:
        print("chosen network is convnext", flush=True)
        for name,param in net.module._net.named_parameters():
            if 'features.7.2' in name: 
                params_to_train.append(param)
            elif 'stage4_reducer' in name:
                params_to_train.append(param)
            elif 'features.7' in name or 'features.6' in name:
                params_to_freeze.append(param)
            # CUDA MEMORY ISSUES? COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
            elif 'features.5' in name or 'features.4' in name:
                params_backbone.append(param)
            else:
                param.requires_grad = False
            # else:
            #     params_backbone.append(param)
    elif 'dinov2_vits14' in args.net:
        print(f"chosen network is {args.net}", flush=True)
        for name,param in net.module._net.named_parameters():
            if ('blocks.11' in name) or ('norm.weight' in name) or ('norm.bias' in name):
                params_to_train.append(param)
            elif ('blocks.10' in name) or ('blocks.9' in name):
                params_to_freeze.append(param)
            else:
                params_backbone.append(param)
    else:
        print("Network is not implemented.", flush=True)     

    classification_weight = []
    classification_bias = []
    for attr in dir(net.module):
        if attr.endswith('_classification'):
            for name, param in getattr(net.module, attr).named_parameters():
                # breakpoint()
                if 'weight' in name:
                    classification_weight.append(param)
                elif 'multiplier' in name:
                    param.requires_grad = False 
                else:
                    if args.bias:
                        classification_bias.append(param)

    proto_presence_weights = []
    for attr in dir(net.module):
        if attr.endswith('_proto_presence'):
            # breakpoint() # type(param) # type(getattr(net.module, attr))
            proto_presence_weights.append(getattr(net.module, attr))

    
    paramlist_net = [
            {"params": params_backbone, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
            {"params": params_to_freeze, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay}]
            # {"params": net.module._add_on.parameters(), "lr": args.lr_block*10., "weight_decay_rate": args.weight_decay}]
    for attr in dir(net.module):
        if attr.endswith('_add_on'):
            paramlist_net.append({"params": getattr(net.module, attr).parameters(), "lr": args.lr_block*10., "weight_decay_rate": args.weight_decay})
            
    paramlist_classifier = [
            {"params": classification_weight, "lr": args.lr, "weight_decay_rate": args.weight_decay},
            {"params": classification_bias, "lr": args.lr, "weight_decay_rate": 0},

            {"params": proto_presence_weights, "lr": args.lr, "weight_decay_rate": args.weight_decay},
    ]
    
    
    if args.optimizer == 'Adam':
        optimizer_net = torch.optim.AdamW(paramlist_net,lr=args.lr,weight_decay=args.weight_decay)
        optimizer_classifier = torch.optim.AdamW(paramlist_classifier,lr=args.lr,weight_decay=args.weight_decay)
        return optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone
    else:
        raise ValueError("this optimizer type is not implemented")

