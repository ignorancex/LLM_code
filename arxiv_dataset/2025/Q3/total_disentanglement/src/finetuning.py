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
from dataset import FinetuningDataset, load_finetuning_data, compute_font_features, compute_char_features
import cv2
from tqdm import tqdm
import pandas as pd
from itertools import combinations, permutations
import csv
import argparse


def main(args):
    # Set up directories
    os.makedirs('checkpoints/{}'.format(args.save_folder), exist_ok=True)
    
    # Save parameters
    params = [
        'NOTE:{}'.format(args.note),
        'DEVICE={}'.format(args.device),
        'PRETRAIN_FOLDER={}'.format(args.pretrain_folder),
        'SAVE_FOLDER={}'.format(args.save_folder),
        'DATASET_DIR={}'.format(args.dataset_dir),
        'SEED={}'.format(args.seed),
        'CHAR_NUM={}'.format(args.char_num),
        'BATCH_SIZE={}'.format(args.batch_size),
        'ZDIM={}'.format(args.zdim),
        'NUM_EPOCHS={}'.format(args.num_epochs),
        'IMG_SIZE={}'.format(args.img_size),
        'WEIGHT_FONT={}'.format(args.w_f),
        'WEIGHT_CHAR={}'.format(args.w_c),
        'WEIGHT_CHAR_CLASS={}'.format(args.w_class),
        'WEIGHT_REC={}'.format(args.w_rec),
    ]
    
    with open('checkpoints/{}/param.txt'.format(args.save_folder), mode='w') as f:
        f.write('\n'.join(params))
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    print(args.zdim)
    
    # Initialize model
    model = DISENTANGLE_MODEL(args.zdim, args.char_num, args.batch_size, args.device, args.img_size).to(args.device)
    if len(args.pretrain_model) > 0:
        print('===pre-train===')
        # Check if pretrain_model is already a full path
        if '/' in args.pretrain_model:
            model_path = args.pretrain_model
        else:
            model_path = 'checkpoints/{}/{}'.format(args.pretrain_folder, args.pretrain_model)
        print(f'Loading pretrained model from: {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=args.device))
    
    # Prepare training data
    random.seed(0)
    ch_list = []
    start = ord('A')
    for i in range(args.char_num):
        ch_list.append(chr(i + start))
    
    # Load training data
    print("Loading training data...")
    img_train, ch_train, font_train, train_font_list = load_finetuning_data(
        args.dataset_dir, ch_list, split='train', img_size=args.img_size
    )
    TRAIN_NUM = len(train_font_list) - 1
    
    # Compute font and character features for training data
    print("Computing font features for training data...")
    font_feat_train, _ = compute_font_features(
        model, args.dataset_dir, ch_list, 'train', args.img_size, args.device
    )
    
    print("Computing character features for training data...")
    ch_feat_train = compute_char_features(
        model, args.dataset_dir, ch_list, 'train', args.img_size, args.device
    )
    
    # Load test data
    print("Loading test data...")
    img_test, ch_test, font_test, valid_font_list = load_finetuning_data(
        args.dataset_dir, ch_list, split='valid', img_size=args.img_size
    )
    TEST_NUM = len(valid_font_list) - 1
    
    # Compute font and character features for test data
    print("Computing font features for test data...")
    font_feat_test, _ = compute_font_features(
        model, args.dataset_dir, ch_list, 'valid', args.img_size, args.device
    )
    
    print("Computing character features for test data...")
    ch_feat_test = compute_char_features(
        model, args.dataset_dir, ch_list, 'valid', args.img_size, args.device
    )
    
    # Create data loaders
    trans1 = torchvision.transforms.ToTensor()
    train_dataset = FinetuningDataset(img_train, ch_train, font_train, transform=trans1)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_dataset = FinetuningDataset(img_test, ch_test, font_test, transform=trans1)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
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
                          'font_class_loss', 'rec_loss1', 'rec_loss2', 'font_mse', 'char_mse',
                          'test_char_loss', 'test_rec_loss1', 'test_rec_loss2', 'test_font_mse', 'test_char_mse']])
    
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
        
        for i, (images, labels, fonts) in enumerate(train_loader):
            # Reconstruction images
            # Encode images
            x = images.to(args.device, torch.float32)
            labels = labels.to(args.device)
            fonts = fonts.to(args.device)
            z_c, z_f, output_c = model.encode(x)
            
            # Character features (font average for each character)
            for j, label in enumerate(labels):
                c_ave = ch_feat_train[label.item()]
                c_feat = torch.cat((c_feat, c_ave), 0) if j != 0 else c_ave
            
            # Font features (character average for each font)
            for j, font in enumerate(fonts):
                f_ave = font_feat_train[font.item()]
                f_feat = torch.cat((f_feat, f_ave), 0) if j != 0 else f_ave
            
            z = torch.cat((z_c.to(args.device), z_f.to(args.device)), axis=1)
            y = model.decode(z)
            
            # Compute loss
            loss_c = cre_loss(output_c, labels)
            loss_rec = l1_loss(y, x)
            loss_c_mse = mse_loss(z_c, c_feat)
            loss_f_mse = mse_loss(z_f, f_feat)
            loss = args.w_rec * loss_rec + args.w_f * loss_f_mse + args.w_c * loss_c_mse + args.w_class * loss_c
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            char_class_loss += loss_c.item() * len(labels)
            font_class_loss += 0
            rec_loss += loss_rec.item() * x.shape[0]
            rec_loss2 += 0
            font_mse += loss_f_mse.item() * x.shape[0]
            char_mse += loss_c_mse.item() * x.shape[0]
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_loss = 0
            test_char_loss = 0
            test_rec_loss = 0
            test_rec_loss2 = 0
            test_font_mse = 0
            test_char_mse = 0
            for i, (images, labels, fonts) in enumerate(test_loader):
                # Reconstruction images
                # Encode images
                x = images.to(args.device, torch.float32)
                labels = labels.to(args.device)
                fonts = fonts.to(args.device)
                z_c, z_f, output_c = model.encode(x)
                
                # Character features
                for j, label in enumerate(labels):
                    c_ave = ch_feat_test[label.item()]
                    c_feat = torch.cat((c_feat, c_ave), 0) if j != 0 else c_ave
                
                # Font features
                for j, font in enumerate(fonts):
                    f_ave = font_feat_test[font.item()]
                    f_feat = torch.cat((f_feat, f_ave), 0) if j != 0 else f_ave
                
                z = torch.cat((z_c.to(args.device), z_f.to(args.device)), axis=1)
                y = model.decode(z)
                
                # Compute loss
                loss_c = cre_loss(output_c, labels)
                loss_rec = l1_loss(y, x)
                loss_c_mse = mse_loss(z_c, c_feat)
                loss_f_mse = mse_loss(z_f, f_feat)
                loss = args.w_rec * loss_rec + args.w_f * loss_f_mse + args.w_c * loss_c_mse + args.w_class * loss_c
                
                test_loss += loss.item() * x.shape[0]
                test_char_loss += loss_c.item() * len(labels)
                test_rec_loss += loss_rec.item() * x.shape[0]
                test_rec_loss2 += 0
                test_font_mse += loss_f_mse.item() * x.shape[0]
                test_char_mse += loss_c_mse.item() * x.shape[0]
        
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
                              font_mse / len(train_dataset), char_mse / len(train_dataset),
                              test_char_loss / len(test_dataset), test_rec_loss / len(test_dataset),
                              test_rec_loss2 / len(test_dataset),
                              test_font_mse / len(test_dataset), test_char_mse / len(test_dataset)
                              ]])
        
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
    parser = argparse.ArgumentParser(description='Finetuning Script')
    
    # Model parameters
    parser.add_argument('--zdim', type=int, default=256, help='Dimension of latent space')
    parser.add_argument('--char_num', type=int, default=26, help='Number of characters')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--early_stopping', type=bool, default=False, help='Enable early stopping')
    
    # Dataset parameters
    parser.add_argument('--dataset_dir', type=str, default='sample_data', help='Path to dataset directory')
    
    # Loss weights
    parser.add_argument('--w_f', type=float, default=1, help='Font loss weight')
    parser.add_argument('--w_c', type=float, default=1, help='Character loss weight')
    parser.add_argument('--w_class', type=float, default=0.001, help='Classification loss weight')
    parser.add_argument('--w_rec', type=float, default=1, help='Reconstruction loss weight')
    
    # Device and paths
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--pretrain_model', type=str, default='best_model.pth', help='Pretrained model name')
    parser.add_argument('--pretrain_folder', type=str, default=None, help='Pretrained model folder')
    parser.add_argument('--save_folder', type=str, default=None, help='Save folder name')
    parser.add_argument('--note', type=str, default='MSE(img),norm', help='Notes about the experiment')
    
    args = parser.parse_args()
    
    # Set defaults based on other parameters if not specified
    if args.pretrain_folder is None:
        args.pretrain_folder = 'pretraining'
    if args.save_folder is None:
        args.save_folder = 'finetuning'
    
    main(args)