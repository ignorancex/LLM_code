
import time
import torch
from dataset import Data
from nodes import Node
from args import args_parser
from utils import *
import numpy as np
import os
import torch.nn as nn
import copy
import torch.optim as optim
import torch.nn.functional as F
import math
from server_funct import *
import wandb
from client_funct import *
from utils import EarlyStopping

if __name__ == '__main__':

    args = args_parser()
    setup_seed(args.random_seed)

    # get ip address, which can help us to know which machine the experiment is done on
    ip_address = get_local_ip()
    args.ip_address = ip_address

    args.iid = 0
    args.noniid_type = 'dirichlet'
    if args.server_method == 'finetune':
        args.server_epochs = 2
    else:
        args.server_epochs = 20

    if args.dataset == 'cifar10':
        args.server_valid_ratio = 0.02
    elif args.dataset == 'cifar100':
        pass

    if args.client_method == 'feddyn':
        args.mu = 0.01
    elif args.client_method == 'fedprox':
        args.mu = 0.01

    # if plus, add sam and lc balanced loss
    if args.client_method == 'multi_step_group_connectivity_plus':
        args.sam = True
        args.is_balanced_loss = True


    setting_name = f'{args.client_method}-{args.server_method}_{args.dataset}_{args.local_model}'
    if 'plus' not in args.client_method:
        if args.sam:
            setting_name += '_sam'
        if args.is_balanced_loss:
            setting_name += '_balanced_loss'
    if args.exp_name is not None:
        setting_name += f'_{args.exp_name}'

    root_path = '/home/FL_group_linear_connectivity'
    output_path = 'fl_outputs/1008/'
    result_path = 'fl_results/1008/'
    save_path_root = root_path

    wandb.init(
        config = args,
        project = 'FL_group_connectivity',
        name = setting_name,
        group=args.group
    )

    torch.cuda.set_device('cuda:'+args.device)

    data = Data(args)

    sample_size = []
    for i in range(args.node_num): 
        sample_size.append(len(data.train_loader[i]))
    size_weights = [i/sum(sample_size) for i in sample_size]
    print('size-based weights',size_weights)

    # initialize the central node
    central_node = Node(-1, data.test_loader[0], data.test_set, args)
    wandb.watch(central_node.model, log='parameters')


    # initialize the client nodes
    client_nodes: Dict[int, Node] = {}
    for i in range(args.node_num): 
        client_nodes[i] = Node(i, data.train_loader[i], data.train_set, args) 
        if args.server_method == 'none':
            client_nodes[i].validate_set = central_node.local_data
            print("set client node validate set to test set")
    
    final_test_acc_recorder = RunningAverage()
    test_acc_recorder = []

    if args.select_ratio == 1.0:
        select_list_recorder = [[i for i in range(args.node_num)] for _ in range(args.T)]
    else:
        select_list_recorder = torch.load(f"results/select_list/num{args.node_num}_ratio{args.select_ratio}_select_list_recorder.pth")



    central_node.global_model_list = []
    interval = 1
    if args.server_method == 'none':
        interval = 20
    earlystopper = EarlyStopping(10)
    best_metric = 0
    for rounds in range(args.T):
        print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1))
        print(setting_name)
        lr_scheduler(rounds, client_nodes, args)

        select_list = select_list_recorder[rounds]


        if rounds == 0:
            central_node.init_model = copy.deepcopy(central_node.model)
        
        #TODO store the global model list
        central_node.global_model_list.append(copy.deepcopy(central_node.model))

        # only store the last `group_model_num` models to save memory
        central_node.global_model_list = central_node.global_model_list[-args.group_model_num:]


        #TODO sample random global models
        if args.client_method == 'curr_round_group_connectivity':
            central_node = generate_global_model_set(args, central_node, client_nodes, select_list, size_weights)


        client_nodes, train_loss = Client_update(args, client_nodes, central_node, select_list)
        if rounds % interval == 0 or rounds >= args.T - 5 :
            avg_client_acc = Client_validate(args, client_nodes, select_list)
        else:
            avg_client_acc = None
        print(args.server_method + args.client_method + ', averaged clients acc is ', avg_client_acc)
        
        # TODO change this partial select function
        central_node = Server_update(args, central_node, client_nodes, select_list, size_weights)
        if args.server_method == 'none':
            acc = avg_client_acc
        else:
            acc = validate(args, central_node, which_dataset = 'local') # test on test datasets, not every client's validate dataset
        print(args.server_method + args.client_method + ', global model test acc is ', acc)
        test_acc_recorder.append(acc)

        if rounds >= args.T - 5:
            final_test_acc_recorder.update(acc)

        try:
            wandb.log({'trainloss': train_loss}, step = rounds)
            if acc is not None:
                wandb.log({'testacc': acc}, step = rounds)
            if avg_client_acc is not None:
                wandb.log({'validacc': avg_client_acc}, step=rounds)
        except:
            pass
            
        if acc is not None:
            best_metric = max(best_metric, acc)
    
    try:
        if "glue" in args.dataset:
            final_test_acc = best_metric
        else:
            final_test_acc = final_test_acc_recorder.value()
        wandb.log({'final_testacc': final_test_acc})
    except:
        pass
