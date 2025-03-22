import os
import argparse
import torchvision
import pandas as pd
import torch 
import numpy as np
import time
import pprint
import tqdm
import exp_configs
from src import datasets, models, metrics
from load_args import load_args
from load_optim import load_optim
from torch.optim.lr_scheduler import ReduceLROnPlateau





def trainval(args, device, exp_dict, datadir, metrics_flag=True):
    # bookkeeping
    # ---------------


    # Set the ramdom seed for reproducibility.
    if args.reproducible:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            if device != torch.device("cpu"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    # set seed
    # ---------------
    seed = 42 + args.run
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
       

    # Dataset
    # -----------

    # Load Train Dataset
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     train_flag=True,
                                     datadir=datadir,
                                     exp_dict=exp_dict)

    train_loader = torch.utils.data.DataLoader(train_set,
                              drop_last=True,
                              shuffle=True,
                              batch_size=exp_dict["batch_size"])

    # Load Val Dataset
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                   train_flag=False,
                                   datadir=datadir,
                                   exp_dict=exp_dict)


    # Model
    # -----------
    model = models.get_model(exp_dict["model"],
                             train_set=train_set).cuda()
    # Choose loss and metric function
    loss_function = metrics.get_metric_function(exp_dict["loss_func"])

    # Load Optimizer
    n_batches_per_epoch = len(train_set)/float(exp_dict["batch_size"])


    opt = load_optim(params=model.parameters(),
                           optim_method=args.optim_method,
                           eta0=args.eta0,
                           alpha=args.alpha,
                           c=args.c,
                           milestones=args.milestones,
                           T_max=args.train_epochs*len(train_loader),
                           n_batches_per_epoch=len(train_loader),
                           nesterov=args.nesterov,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           gamma=args.gamma,
                           coeff=args.coeff)
   
    if args.optim_method == 'SGD_ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(opt,
                                      mode='min',
                                      factor=args.alpha,
                                      patience=args.patience,
                                      threshold=args.threshold)
    
    
    # Train & Val
    # ------------

    all_train_losses = []
    all_train_accuracies = []
    all_test_losses = []
    all_test_accuracies = []
    all_grad_norm=[]
    all_step_length=[]
    counter=-1
    
    for epoch in range(1, args.train_epochs + 1):
        model.train()
        counter=counter+1
        for data in train_loader:
            inputs, labels = data
            if args.dataset == 'Flower102':
                labels = torch.sub(labels, 1)
            inputs, labels = inputs.to(device), labels.to(device)

            opt.zero_grad()        

            if args.optim_method.startswith('SLS'):
                closure = lambda : loss_function(model, inputs, labels, backwards=False)
                optimizer.step(closure)
                
            elif args.optim_method.startswith('SCGS'):
                closure = lambda model : loss_function(model, inputs, labels, backwards=False)
                l,g=opt.step(model,counter,closure) 
                all_grad_norm.append(g)
                
            elif args.optim_method.startswith('SCGWSA'):
                outputs = model(inputs)
                closure = lambda : loss_function(outputs, labels)
                opt.step(model,counter,closure)  
            else:
                outputs = model(inputs)
                loss =loss_function(model, inputs, labels, backwards=False)
                loss.backward()
                if 'Polyak' in args.optim_method:
                    opt.step(loss.item())
                else:
                    opt.step()

        # Evaluate the model on training and validation dataset.
        if args.optim_method == 'SGD_ReduceLROnPlateau' or (epoch % args.eval_interval == 0):
        
            train_loss = metrics.compute_metric_on_dataset(model, train_set,metric_name=exp_dict["loss_func"])
          
            test_accuracy  = metrics.compute_metric_on_dataset(model, val_set, metric_name=exp_dict["acc_func"])
                                                                                                                
            all_train_losses.append(train_loss)
            all_test_accuracies.append(test_accuracy)

            print('Epoch %d --- ' % (epoch),
                  'train: loss - %g, ' % (train_loss),
                  'accuracy - %g' % (test_accuracy))
                  
                  
           # if args.optim_method == 'SGD_ReduceLROnPlateau':
                #scheduler.step(test_loss)


        
    print("Evaluating...")
    final_train_loss = metrics.compute_metric_on_dataset(model, train_set,metric_name=exp_dict["loss_func"])
          
    final_test_accuracy  = metrics.compute_metric_on_dataset(model, val_set, metric_name=exp_dict["acc_func"])
    
    # Logging results.
    print('Writing the results.')
    if not os.path.exists(args.log_folder):
            os.makedirs(args.log_folder)
    log_name = (('%s_%s_' % (args.dataset, args.optim_method))
                     + ('Eta0_%g_' % (args.eta0))
                     + ('WD_%g_' % (args.weight_decay))
                     + (('Mom_%g_' % (args.momentum))
                         if args.optim_method.startswith('SGD') else '')
                     + (('alpha_%g_' % (args.alpha))
                         if args.optim_method not in ['Adam', 'SGD'] else '')
                     + (('Milestones_%s_' % ('_'.join(args.milestones)))
                         if args.optim_method == 'SGD_Stage_Decay' else '')
                     + (('c_%g_' % (args.c))
                         if args.optim_method.startswith('SLS') else '')
                     + (('Patience_%d_Thres_%g_' % (args.patience, args.threshold))
                         if args.optim_method == 'SGD_ReduceLROnPlateau' else '')
                     + ('Epoch_%d_Batch_%d_' % (args.train_epochs, args.batchsize))
                     + ('%s' % ('Validation' if args.validation else 'Test'))
                     + '.txt')
    mode = 'w' if args.validation else 'a'
    with open(args.log_folder + '/' + log_name, mode) as f:
            f.write('Training running losses:\n')
            f.write('{0}\n'.format(all_train_losses))
            f.write('Final training loss is %g\n' % final_train_loss)

            f.write('Test running accuracies:\n')
            f.write('{0}\n'.format(all_test_accuracies))               
            f.write('Final test accuracy is %g\n' % final_test_accuracy) 
            
            #f.write('grad norms:\n')
            #f.write('{0}\n'.format(all_grad_norms))

    print('Finished.')
        



if __name__ == '__main__':

    args = load_args()
    
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
        

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment  
        
        exp_list = [exp_dict]
        
    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]


    # Run experiments
    # ----------------------------
    for exp_dict in exp_list:
        # do trainval
        trainval(args, device, exp_dict=exp_dict,
                datadir=args.datadir)
