import numpy as np
import time
import gc

import os
os.environ['DGLBACKEND'] = 'pytorch'  # tell DGL what backend to use

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from utils_residual_gnn import set_seed, seed_worker, set_device, reset_weights, myoptimizer, Criterion
from utils_residual_gnn import train_residual_vgg, evaluate_metric, model_size
from utils_residual_gnn import save_checkpoint, load_checkpoint
from utils_residual_gnn import features_collate4metric as collate

from data_loading import KinFeaturetDataset
from GNN_Residual_VGG import GNN_Residual_VGG
import csv
import settings as st
cnt = 0
"""change made by OMiD"""
def write_details(path, fold_num, epoch, train_acc, test_acc, my_loss, BCE, MSEP, MSEN, MSEC, triplet, cross):
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        to_write = []
        
        to_write.append(f'{fold_num}')
        to_write.append(f'{epoch: .0f}')
        
        to_write.append(f'{train_acc*100: .1f}')
        to_write.append(f'{test_acc*100: .2f}')
        
        to_write.append(f'{my_loss: .1f}')
        to_write.append(f'{BCE: .3f}')
        to_write.append(f'{MSEP: .3f}')
        to_write.append(f'{MSEN: .3f}')
        to_write.append(f'{MSEC: .3f}')
        to_write.append(f'{triplet: .3f}')
        to_write.append(f'{cross: .2f}')
        
        
        writer.writerow(to_write)
    file.close()
    
def mean_max_acc(path, printt=False, results=False):
    # top_acc_of_each_fold    
    max_test_accs = [0, 0, 0, 0, 0]
    # (first_occurance_of_top_acc, last_occurance_of_top_acc)
    max_test_accs_epochs = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    
    with open(path, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
            fold_num, test_acc = int(row[0]), float(row[3]) 
            if max_test_accs[fold_num-1] < test_acc:
                max_test_accs[fold_num-1] = test_acc
                max_test_accs_epochs[fold_num-1][0] = int(row[1])
                max_test_accs_epochs[fold_num-1][1] = int(row[1])
            
            elif max_test_accs[fold_num-1] == test_acc:
                max_test_accs_epochs[fold_num-1][1] = int(row[1])

    file.close()
    
    if printt:
        print(max_test_accs, '\t', max_test_accs_epochs, '\t\t', sum(max_test_accs)/5)
    
    if results:
        return max_test_accs
        
    return (sum(max_test_accs)/5)

def mean_dataset(ls, dir, save=True):
    accs = sum([mean_max_acc(os.path.join(dir, 'detail_'+i+'.csv')) for i in ls])
    print('Dataset Accuracy =', accs/len(ls))  

    if save:
        path = os.path.join(dir, f'Final Accuracy {accs/len(ls):.2f}.txt')

        accs = [mean_max_acc(os.path.join(dir, 'detail_'+i+'.csv'), results=True) for i in ls]

        with open(path, 'w') as file:
            for i in range(len(ls)):
                file.write(ls[i]+'\t'+str(accs[i])+'\t'+str(sum(accs[i])/len(accs[i]))+'\n') 

        file.close()
    
def save_config(dir):
    lines = [
            f'Epochs           :\t{st.EPOCHS}\n',
            f'Batchsize        :\t{st.BATCH_SIZE}\n',
            f'CCLstartingweight:\t{st.CCL_STARTING_WEIGHT}\n',
            f'CCLdecayrate     :\t{st.config[st.I]["CCL_DECAY_RATE"]}\n',
            f'Bcestartingweight:\t{st.BCE_STARTING_WEIGHT}\n',
            f'Bceweightdecay   :\t{st.BCE_WEIGHT_DECAY}\n',
            f'Kickinepoch1     :\t{st.KICK_IN_EPOCH}\n',
            f'Kickinepoch2     :\t{st.KICK_IN_EPOCH2}\n',
            f'FC1              :\t{st.config[st.I]["FC1"]}\n',
            f'FC2              :\t{st.config[st.I]["FC2"]}\n',
            f'Seperate         :\t{st.config[st.I]["SEPERATE_RUN"]}\n',
            f'Outputdim        :\t{st.config[st.I]["OUTPUTDIM"]}\n',
            f'Order            :\t{st.ORDER}\n',
    ]
    path = os.path.join(dir, f'config{st.I}.txt')
    with open(path, 'w') as file:  
        file.writelines(lines)  

# Function to train a model for one fold
def train_fold(fold_num, model, trn_dataloader, tst_dataloader, optimizer, scheduler, n_epochs, criterion, DEVICE, detail_path):

    """Change made by omid"""
    bce_weight = st.BCE_STARTING_WEIGHT        # 1
    ccl_weight = st.CCL_STARTING_WEIGHT     # 0.01
    
    bce_decay_rate = st.BCE_WEIGHT_DECAY    # 1  
    ccl_decay_rate = st.config[st.I]["CCL_DECAY_RATE"]  #1.1 gave 0.95
    
    
    for epoch in range(n_epochs):
        model.isTrain = True
        start = time.time()
        train_loss, train_acc, my_loss, bce, msep, msen, msec, triplet, cross = train_residual_vgg(
            model, trn_dataloader, criterion, optimizer, DEVICE, ccl_weight, bce_weight, epoch)
        model.isTrain = False
        print(f'\r{time.time() - start:.3f}', end='', flush=True)
        
        # Test model
        test_loss, test_acc = evaluate_metric(model, tst_dataloader, criterion, DEVICE)
        scheduler.step()

        # Write details to CSV
        write_details(detail_path, fold_num, epoch+1, train_acc, test_acc, my_loss, bce, msep, msen, msec, triplet, cross)

        bce_weight *= bce_decay_rate
        ccl_weight *= ccl_decay_rate
        global cnt
        if cnt==0:
            os.system('cls')
            cnt +=1 

        # Optionally save the model if needed
        # if test_acc > max_acc:
        #     max_acc = test_acc
        #     FILENAME = f'model_fold_{fold_num}_epoch_{epoch+1}_acc_{test_acc:.2f}.pt'
        #     SAVE_PATH = os.path.join('CheckPoints', FILENAME)
        #     save_checkpoint(model, optimizer, SAVE_PATH, epoch, 'kin_type', batch_size, fold_num)

    # print(f'Finished training for fold {fold_num}')
     
def run_training_for_folds(dir):
    # Folds to train on

    n_folds = 5
    ls = ['father_dau', 'father_son', 'mother_dau', 'mother_son']
    
    # Parameters
    n_epochs = st.EPOCHS  # or 60, based on your choice
    batch_size = st.BATCH_SIZE
    input_dim = 2688#1152#3200#4736#2688
    output_dim = 1
    
        # Evaluate the accuracy across folds
    # mean_dataset(ls, dir)
    for csv_metadata_file_path in ls:
        csv_metadata_file = csv_metadata_file_path + '.csv'
        detail_path = 'detail_' + csv_metadata_file
        detail_path = os.path.join(dir, detail_path)

        # mean_max_acc(detail_path, printt=True)
        # continue

        
        # Write header to detail file
        with open(detail_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            data = ['fold_num', 'epoch', 'train_acc', 'test_acc', 'my_loss', 'bce', 'msep', 'msen', 'msec', 'triplet', 'cross']
            writer.writerow(data)

        # Prepare data paths
        data_path = os.path.join(r'D:\Uni\Term 9\Project\KIN_I\KinFaceW-I')
        csv_metadata_file_fullpath = os.path.join(data_path, csv_metadata_file)
        #features_file = 'eff-vit-resnet-residual-features-kinfacewi-2688.npz'
        #features_file = 'eff-vit-vggface2-resnet-residual-features-kinfacewi-4736.npz'
        #features_file = 'facenet-resnet-residual-kinfacewi-1152.npz'
        #features_file = 'resnet-residual-facenet-kinfacewi-1152.npz'
        #features_file = 'features-vggface2-resnet-residual_2688.npz'
        #features_file = 'vggface2-resnet-residual-kinfacewi-2688.npz'
        # features_file = "merged-resnet64-resnet1024-kinfacewi-1536.npz"
        # features_file = 'merged-resnet64-resnet1024-vggface1024-kinfacewi-3584.npz'
        features_file = 'features-vggface2-resnet-residual_1024_64_2688.npz'
        features_file_fullpath = os.path.join(data_path, features_file)

        if not st.config[st.I]["SEPERATE_RUN"]:
            set_seed(20240804)
        for fold_num in st.ORDER:
            if st.config[st.I]["SEPERATE_RUN"]:
                set_seed(seed=20240804)
            DEVICE = set_device()
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()

            # Reset model, optimizer, and scheduler for each fold
            model = GNN_Residual_VGG(input_dim, output_dim).to(DEVICE)
            # torch.save(model.state_dict(), "model.pth")
            # model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            # print(f"Estimated model size: {model_size_mb:.2f} MB")
            # print(model_size(model))
            # reset_weights(model)                                            # 1e-4
            optimizer = myoptimizer(model, {'lr': 1e-4, 'weight_decay': 0})
            lr_milestones = [ 20, 40, 60,90, 120, 150, 200]
            lr_decay = 0.5 ## learning rate decay
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.5)
            criterion = Criterion()

            # Load datasets
            trn_set = KinFeaturetDataset(csv_metadata_file_fullpath, features_file_fullpath, isTrain=True, fold_num=fold_num)
            tst_set = KinFeaturetDataset(csv_metadata_file_fullpath, features_file_fullpath, isTrain=False, fold_num=fold_num)

            trn_dataloader = DataLoader(trn_set, batch_size=batch_size, shuffle=True, collate_fn=collate, worker_init_fn=seed_worker)
            tst_dataloader = DataLoader(tst_set, batch_size=batch_size, shuffle=False, collate_fn=collate, worker_init_fn=seed_worker)

            # Train the model for this fold
            train_fold(fold_num, model, trn_dataloader, tst_dataloader, optimizer, scheduler, n_epochs, criterion, DEVICE, detail_path)
            
            
            # Delete model and related objects after fold training
            del model
            del optimizer
            del scheduler
            del trn_dataloader
            del tst_dataloader
            del criterion

            gc.collect()
            torch.cuda.empty_cache()
            
            
        mean_max_acc(detail_path, printt=True)
        
    # Evaluate the accuracy across folds
    mean_dataset(ls, dir)

def run_all_configs():
    while st.I < len(st.config.keys()):
        start = time.time()

        dir = 'Results/'
        dir += f'ccl_decay-{st.config[st.I]["CCL_DECAY_RATE"]}_'
        dir += f'fc1-{st.config[st.I]["FC1"]}_'
        dir += f'fc2-{st.config[st.I]["FC2"]}_'
        dir += f'sep-{st.config[st.I]["SEPERATE_RUN"]}_'
        dir += f'outdim-{st.config[st.I]["OUTPUTDIM"]}/'
        os.makedirs(dir, exist_ok=True)

        save_config(dir)

        run_training_for_folds(dir)

        st.I += 1
        
        end = time.time()
        print(f'It took {end-start} seconds!!')


if __name__ == '__main__':
    run_all_configs()  
    











"""
print(f'Epochs           :\t{st.EPOCHS}')
print(f'Batchsize        :\t{st.BATCH_SIZE}')
print(f'CCLstartingweight:\t{st.CCL_STARTING_WEIGHT}')
print(f'CCLdecayrate     :\t{st.config[st.I]["CCL_DECAY_RATE"]}')
print(f'Bcestartingweight:\t{st.BCE_STARTING_WEIGHT}')
print(f'Bceweightdecay   :\t{st.BCE_WEIGHT_DECAY}')
print(f'Kickinepoch1     :\t{st.KICK_IN_EPOCH}')
print(f'Kickinepoch2     :\t{st.KICK_IN_EPOCH2}')
print(f'FC1              :\t{st.config[st.I]["FC1"]}')
print(f'FC2              :\t{st.config[st.I]["FC2"]}')
print(f'Seperate         :\t{st.config[st.I]["SEPERATE_RUN"]}')
print(f'Outputdim        :\t{st.config[st.I]["OUTPUTDIM"]}')
print(f'Order            :\t{st.ORDER}')
"""