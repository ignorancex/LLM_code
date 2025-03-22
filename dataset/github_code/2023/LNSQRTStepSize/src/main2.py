if __name__ == "__main__":
    import torch
    import torch.nn as nn

    import numpy as np
    import os
    import random

    from load_args import load_args
    from data_loader import data_loader
    from mnist_cnn import MNISTConvNet
    from cifar_cnn import CIFARConvNet
    from flower_cnn import FLOWERConvNet
    from cifar10_resnet import resnet20
    from cifar100_densenet import densenet
    from train import train
    from evaluate import evaluate

    def main():
        args = load_args()
        
        for dataset in ['FashionMNIST','CIFAR10', 'CIFAR100']:
            args.dataset=dataset
            print(args.dataset)
            for a in [0.015,0.02,0.025,0.035,0.04,0.045,0.055]:
            
                args.eta0=0.05
                print(args.eta0)
                args.alpha=a
                print(args.alpha)
            
                              
                sum_final_train_loss=0
                sum_final_train_accuracy=0
                sum_final_test_loss=0
                sum_final_test_accuracy=0
                
                total_final_train_loss=[]
                total_final_train_accuracy=[]
                total_final_test_loss=[]
                total_final_test_accuracy=[]
                
                if not os.path.exists(args.log_folder):
                    os.makedirs(args.log_folder)
        
                log_name_total = (('%s_%s_' % (args.dataset, args.optim_method))
                     + ('alpha_%g_' % (args.alpha))
                     +('total')
                     + '.txt')
        
            
                for run_number in [1,2,3,4,5]:
                    
                    # Check the availability of GPU.
                    use_cuda = args.use_cuda and torch.cuda.is_available()
                    device = torch.device("cuda:0" if use_cuda else "cpu")

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
                    #seed = 42 + run_number
                    seed = 43
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
        
                    print ("seed")
                    print(seed)
                    #----------------------------train epoches----------------------
                    
                    if args.dataset == 'FashionMNIST':
                         weight_decay=0.0001 
                         train_epochs=100 
                    if args.dataset == 'CIFAR10':
                         weight_decay=0.0001
                         train_epochs=164                            
                    elif args.dataset =='CIFAR100':
                         weight_decay=0.0005  
                         train_epochs=60    
                            
                    args.weight_decay=weight_decay
                    args.train_epochs=train_epochs
                    
                    # Load data, note we will also call the validation set as the test set.
                    print('Loading data...')
                    dataset = data_loader(dataset_name=args.dataset,
                              dataroot=args.dataroot,
                              batch_size=args.batchsize,
                              val_ratio=(args.val_ratio if args.validation else 0))
                    train_loader = dataset[0]
                    if args.validation:
                        test_loader = dataset[1]
                    else:
                        test_loader = dataset[2]

                    # Define the model and the loss function.
                    if args.dataset == 'CIFAR10':
                          net = resnet20()
                    elif args.dataset == 'CIFAR100':
                          net = densenet(depth=100, growthRate=12, num_classes=100)
                          #net = CIFARConvNet()
                    elif args.dataset in ['MNIST', 'FashionMNIST']:
                          net = MNISTConvNet()
                    else:
                          raise ValueError("Unsupported dataset {0}.".format(args.dataset))    
                    net.to(device)
                    criterion = nn.CrossEntropyLoss()

                    # Train and evaluate the model.
                    print("Training...")
                    running_stats = train(args, train_loader, test_loader, net,
                              criterion, device)
                    all_train_losses, all_train_accuracies = running_stats[:2]
                    all_test_losses, all_test_accuracies, all_grad_norms, all_step_length = running_stats[2:]

                    print("Evaluating...")
                    final_train_loss, final_train_accuracy = evaluate(args,train_loader, net,
                                                          criterion, device)
                    final_test_loss, final_test_accuracy = evaluate(args,test_loader, net,
                                                        criterion, device)

                       # Logging results.
                    print('Writing the results.')
                    if not os.path.exists(args.log_folder):
                      os.makedirs(args.log_folder)
        
                    log_name = (('%s_%s_' % (args.dataset, args.optim_method))
                         + ('Eta0_%g_' % (args.eta0))
                         + ('Run_%g_' % (run_number))
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
                       f.write('Training running accuracies:\n')
                       f.write('{0}\n'.format(all_train_accuracies))
                       f.write('Final training loss is %g\n' % final_train_loss)
                       f.write('Final training accuracy is %g\n' % final_train_accuracy)

                       f.write('Test running losses:\n')
                       f.write('{0}\n'.format(all_test_losses))
                       f.write('Test running accuracies:\n')
                       f.write('{0}\n'.format(all_test_accuracies))               
                       f.write('Final test loss is %g\n' % final_test_loss)
                       f.write('Final test accuracy is %g\n' % final_test_accuracy) 
                       
                    with open(args.log_folder + '/' + log_name_total, mode) as f:
                       f.write('Training running losses:\n')
                       f.write('{0}\n'.format(all_train_losses))
                       f.write('Training running accuracies:\n')
                       f.write('{0}\n'.format(all_train_accuracies))
                       f.write('Final training loss is %g\n' % final_train_loss)
                       f.write('Final training accuracy is %g\n' % final_train_accuracy)

                       f.write('Test running losses:\n')
                       f.write('{0}\n'.format(all_test_losses))
                       f.write('Test running accuracies:\n')
                       f.write('{0}\n'.format(all_test_accuracies))               
                       f.write('Final test loss is %g\n' % final_test_loss)
                       f.write('Final test accuracy is %g\n' % final_test_accuracy) 
                       
                    sum_final_train_loss+=final_train_loss
                    sum_final_train_accuracy+=final_train_accuracy
                    sum_final_test_loss+=final_test_loss
                    sum_final_test_accuracy+=final_test_accuracy
                    
                    total_final_train_loss+=[final_train_loss]
                    total_final_train_accuracy+=[final_train_accuracy]
                    total_final_test_loss+=[final_test_loss]
                    total_final_test_accuracy+=[final_test_accuracy]
                   
                avg_final_train_loss=sum_final_train_loss/5.0
                avg_final_train_accuracy=sum_final_train_accuracy/5.0
                avg_final_test_loss=sum_final_test_loss/5.0
                avg_final_test_accuracy=sum_final_test_accuracy/5.0
                
                std_final_train_loss=np.std(total_final_train_loss)*1.96/2.24
                std_final_train_accuracy=np.std(total_final_train_accuracy)*1.96/2.24
                std_final_test_loss=np.std(total_final_test_loss)*1.96/2.24
                std_final_test_accuracy=np.std(total_final_test_accuracy)*1.96/2.24
            
                #report the mean  final_test_loss and final_test_accuracy for each eta0   
                log_name2 = (('%s_%s_' % (args.dataset, args.optim_method))
                     + ('alpha_%g_' % (args.alpha))
                     + ('Average')
                     + '.txt')
                with open(args.log_folder + '/' + log_name2, 'w' ) as f:
                       f.write('Average training loss is %g\n' % avg_final_train_loss)
                       f.write('Average training accuracy is %g\n' % avg_final_train_accuracy)
             
                       f.write('Average test loss is %g\n' % avg_final_test_loss)
                       f.write('Average test accuracy is %g\n' % avg_final_test_accuracy) 
                       
                       f.write('std training loss is %g\n' % std_final_train_loss)
                       f.write('std training accuracy is %g\n' % std_final_train_accuracy)
             
                       f.write('std test loss is %g\n' % std_final_test_loss)
                       f.write('std test accuracy is %g\n' % std_final_test_accuracy) 
                   

               
        print('Finished.')
        #if not os.path.exists(args.log_step_length_folder):
            #os.makedirs(args.log_sten_length_folder)
        #step_length_log_name = (('%s_%s_' % (args.dataset, args.optim_method))
                   #  + ('Eta0_%g_' % (args.eta0))
                    # + '.txt')
        #mode = 'w' if args.validation else 'a'
        #with open(args.log_step_length_folder + '/' + step_length_log_name, mode) as f:
            #f.write('step_length:\n')
           # f.write('{0}\n'.format(all_step_length))

    main()
