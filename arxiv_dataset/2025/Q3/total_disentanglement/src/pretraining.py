import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import glob
from models import DISENTANGLE_MODEL
from dataset import PretrainingDataset, load_pretraining_data
import cv2
from tqdm import tqdm
import pandas as pd
from itertools import combinations, permutations
import csv
import time
import argparse


def main(args):
    # Set up directories
    os.makedirs('checkpoints/{}'.format(args.save_folder), exist_ok=True)
    
    # Save parameters
    params = [
        'NOTE:{}'.format(args.note),
        'DEVICE={}'.format(args.device),
        'SAVE_FOLDER={}'.format(args.save_folder),
        'DATASET_DIR={}'.format(args.dataset_dir),
        'SEED={}'.format(args.seed),
        'CHAR_NUM={}'.format(args.char_num),
        'BATCH_SIZE={}'.format(args.batch_size),
        'ZDIM={}'.format(args.zdim),
        'NUM_EPOCHS={}'.format(args.num_epochs),
        'IMG_SIZE={}'.format(args.img_size),
        'WEIGHT_CHAR_CLASS={}'.format(args.w_class),
        'WEIGHT_REC={}'.format(args.w_rec),
        'WEIGHT_REC2={}'.format(args.w_rec2),
    ]
    
    with open('checkpoints/{}/param.txt'.format(args.save_folder), mode='w') as f:
        f.write('\n'.join(params))
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    print(args.zdim)
    
    # Prepare training data
    random.seed(0)
    ch_list = []
    start = ord('A')
    for i in range(args.char_num):
        ch_list.append(chr(i + start))
    
    # Load training data
    print("Loading training data...")
    img_dict, img_path_list, font_list, char_list, ch_train, font_train = load_pretraining_data(
        args.dataset_dir, ch_list, split='train', img_size=args.img_size
    )
    
    trans1 = torchvision.transforms.ToTensor()
    
    train_dataset = PretrainingDataset(
        img_dict=img_dict,
        img_path_list=img_path_list,
        font_list=font_list,
        char_list=char_list,
        label=ch_train,
        font=font_train,
        split="train",
        transform=trans1,
        root_path=f'./{args.dataset_dir}'
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    # Load test data
    print("Loading test data...")
    img_dict_test, img_path_list_test, font_list_test, char_list_test, ch_test, font_test = load_pretraining_data(
        args.dataset_dir, ch_list, split='valid', img_size=args.img_size
    )
    
    test_dataset = PretrainingDataset(
        img_dict=img_dict_test,
        img_path_list=img_path_list_test,
        font_list=font_list_test,
        char_list=char_list_test,
        label=ch_test,
        font=font_test,
        split="valid",
        transform=trans1,
        root_path=f'./{args.dataset_dir}'
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    # Initialize model
    model = DISENTANGLE_MODEL(args.zdim, args.char_num, args.batch_size, args.device, args.img_size).to(args.device)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    kld_loss = nn.KLDivLoss()
    bce_loss = nn.BCELoss()
    cre_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    min_test_loss = np.inf
    min_loss = np.inf
    early_flag = 0
    
    dataset_name = os.path.basename(args.dataset_dir)
    with open('checkpoints/{}/{}_train_loss.csv'.format(args.save_folder, dataset_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerows([['epoch', 'train_loss', 'valid_loss', 'char_class_loss',
                          'font_class_loss', 'rec_loss1', 'rec_loss2',
                          'test_char_loss', 'test_rec_loss1', 'test_rec_loss2']])
    
    # Training loop
    for e in range(args.num_epochs):
        train_loss = 0
        char_class_loss = 0
        font_class_loss = 0
        rec_loss = 0
        rec_loss2 = 0
        font_mse = 0
        char_mse = 0
        
        model.train()
        t1 = time.time()
        for i, (images, labels, fonts, pair, gt) in enumerate(train_loader):
            # Reconstruction images
            # Encode images
            x = images.to(args.device, torch.float32)
            labels = labels.to(args.device)
            fonts = fonts.to(args.device)
            pair = pair.to(args.device, torch.float32)
            gt = gt.to(args.device, torch.float32)
            z_c, z_f, output_c = model.encode(x)
            pair_z_c, pair_z_f, _ = model.encode(pair)
            
            z = torch.cat((z_c.to(args.device), z_f.to(args.device)), axis=1)
            y = model.decode(z)
            pair_z = torch.cat((z_c.to(args.device), pair_z_f.to(args.device)), axis=1)
            pair_y = model.decode(pair_z)
            
            # Compute loss
            loss_c = cre_loss(output_c, labels)
            loss_rec = l1_loss(y, x)
            loss_rec2 = l1_loss(pair_y, gt)
            loss = args.w_class * loss_c + args.w_rec * loss_rec + args.w_rec2 * loss_rec2
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            char_class_loss += loss_c.item() * len(labels)
            font_class_loss += 0
            rec_loss += loss_rec.item() * x.shape[0]
            rec_loss2 += loss_rec2.item() * x.shape[0]
            t2 = time.time()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_loss = 0
            test_char_loss = 0
            test_rec_loss = 0
            test_rec_loss2 = 0
            test_font_mse = 0
            test_char_mse = 0
            for i, (images, labels, fonts, pair, gt) in enumerate(test_loader):
                # Reconstruction images
                # Encode images
                x = images.to(args.device, torch.float32)
                labels = labels.to(args.device)
                fonts = fonts.to(args.device)
                pair = pair.to(args.device, torch.float32)
                gt = gt.to(args.device)
                z_c, z_f, output_c = model.encode(x)
                pair_z_c, pair_z_f, _ = model.encode(pair)
                
                z = torch.cat((z_c.to(args.device), z_f.to(args.device)), axis=1)
                y = model.decode(z)
                
                pair_z = torch.cat((z_c.to(args.device), pair_z_f.to(args.device)), axis=1)
                pair_y = model.decode(pair_z)
                
                # Compute loss
                loss_c = cre_loss(output_c, labels)
                loss_rec = l1_loss(y, x)
                loss_rec2 = l1_loss(pair_y, gt)
                loss = args.w_class * loss_c + args.w_rec * loss_rec + args.w_rec2 * loss_rec2
                
                test_loss += loss.item() * x.shape[0]
                test_char_loss += loss_c.item() * len(labels)
                test_rec_loss += loss_rec.item() * x.shape[0]
                test_rec_loss2 += loss_rec2.item() * x.shape[0]
        
        # Early stopping
        if test_loss / len(test_dataset) <= min_test_loss:
            model_path = 'checkpoints/{}/best_model.pth'.format(args.save_folder)
            torch.save(model.state_dict(), model_path)
            min_test_loss = test_loss / len(test_dataset)
            count = 0
            print('Update best epoch {}'.format(e + 1))
        elif (test_loss / len(test_dataset) > min_test_loss) and (args.early_stopping == True):
            count += 1
            if count == 10:
                print('===early stop===')
                break
        else:
            count = 0
        
        if (e + 1) % 5 == 0:
            model_path = 'checkpoints/{}/{}ep_model.pth'.format(args.save_folder, e + 1)
            torch.save(model.state_dict(), model_path)
        
        print(f'epoch: {e + 1} train_loss: {train_loss / len(train_dataset)}')
        print(f'epoch: {e + 1} test_loss: {test_loss / len(test_dataset)}')
        
        with open('checkpoints/{}/{}_train_loss.csv'.format(args.save_folder, dataset_name), 'a') as f:
            writer = csv.writer(f)
            writer.writerows([[e + 1, train_loss / len(train_dataset), test_loss / len(test_dataset),
                              char_class_loss / len(train_dataset), font_class_loss / len(train_dataset),
                              rec_loss / len(train_dataset), rec_loss2 / len(train_dataset),
                              test_char_loss / len(test_dataset), test_rec_loss / len(test_dataset),
                              test_rec_loss2 / len(test_dataset)]])
        
        model_path = 'checkpoints/{}/least_model.pth'.format(args.save_folder)
        torch.save(model.state_dict(), model_path)
    
    # Plot training curves
    df = pd.read_csv('checkpoints/{}/{}_train_loss.csv'.format(args.save_folder, dataset_name))
    loss_list = df.columns.values
    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    for i, loss in enumerate(loss_list):
        if loss == 'epoch':
            x = df['epoch'].values
        else:
            y = df[loss].values
            ax = fig.add_subplot(4, 4, i)
            ax.set_title(loss)
            ax.plot(x, y)
    plt.savefig('checkpoints/{}/{}_loss.png'.format(args.save_folder, dataset_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretraining Script')
    
    # Model parameters
    parser.add_argument('--zdim', type=int, default=256, help='Dimension of latent space')
    parser.add_argument('--char_num', type=int, default=26, help='Number of characters')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--early_stopping', type=bool, default=True, help='Enable early stopping')
    
    # Dataset parameters
    parser.add_argument('--dataset_dir', type=str, default='sample_data', help='Path to dataset directory')
    
    # Loss weights
    parser.add_argument('--w_class', type=float, default=0.001, help='Classification loss weight')
    parser.add_argument('--w_rec', type=float, default=1, help='Reconstruction loss weight')
    parser.add_argument('--w_rec2', type=float, default=1, help='Second reconstruction loss weight')
    
    # Device and paths
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--save_folder', type=str, default="pretraining", help='Save folder name')
    parser.add_argument('--note', type=str, default='', help='Notes about the experiment')
    
    args = parser.parse_args()
    
    main(args)