#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 21:51:54 2020

@author: endiqq
"""


import torch
# from model import Trainer
# from batch_gen import BatchGenerator
import os
import argparse
import random
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transfer_models import Transfer_learning as tf_learning
from models import Xray_Dataset, Dataloader, Trainer, FusionDataset
import pickle
import numpy as np
from Fusion_models import Late_Fusion_Net as latefusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train') # train_unlabel
parser.add_argument('--network', default="Res50")
parser.add_argument('--dataset', default='val_ds') #1.test_ds; 2.additional_test_ds; 3. test_ds_covid_only 4. additional_covid_combine
parser.add_argument('--type', default='CXR')
parser.add_argument('--per_teacher', default = 0.1)


args = parser.parse_args()

#Parameters
K_fold = 5
nepoch = 50
num_classes = 3
batch_size = 32
lr = 1e-5
    
finetune = ['Entire']#[]
percent = ['1']
wts = 'Best'

num_student = 1

K = 0.25
P = 1


Trainer_models = Trainer(lr, num_classes)

if  ( args.action == 'retest'  or args.action == 'retest_based_both' or args.action == 'latefusion_retest_2models'):#latefusion_test= 2 models
    # print('pass')
    
    # Container
    all_avg_single_Dataset_acc = []
    all_var_single_Dataset_acc = []
    all_Dataset_acc = []
    
    overall_all_avg_single_Dataset_acc = []
    overall_all_var_single_Dataset_acc = []
    overall_all_Dataset_acc = []
    
    NN_alltype_all_labels = []
    NN_alltype_all_preds = []
    NN_alltype_all_scores = []
    NN_alltype_all_paths = []
    
    for ff in finetune:
        print (ff)
        print('-' * 30)
        for per in percent:
            print (per)
            print('-' * 30)    
            for iidx, Dataset in enumerate([args.type]):
                print (Dataset)
                print('-' * 30)
                
                if (args.action != 'latefusion_retest_2models'):
                    model_conv, size, pretrained, num_ftrs = tf_learning().def_model(args.network, num_classes, feature_extract = 'fc', use_pretrained=False)
                    embedding_dim = num_ftrs
                    model_conv = model_conv.to(device)
                    
                else:
                    model_conv_1, size, pretrained, num_ftrs = tf_learning().def_model(args.network, num_classes, feature_extract = 'fc', use_pretrained=False)
                    model_conv_2, size, pretrained, num_ftrs = tf_learning().def_model(args.network, num_classes, feature_extract = 'fc', use_pretrained=False)
                    embedding_dim = num_ftrs
                    model_conv_1 = model_conv_1.to(device)
                    model_conv_2 = model_conv_2.to(device)
                                    
                single_Dataset_acc = []
                overall_single_Dataset_acc = []
                
                for k in range(K_fold): 
                    print (k)
                    # Prepare for dataset
                    if (args.action == 'retest' or args.action == 'latefusion_retest_2models' or args.action == 'retest_based_both'):                        
                        test_ds = args.dataset+'.txt'
                    elif args.action == 'test_unlabel_chain':
                        test_ds = str(args.per_teacher)+'_unlabeled_test_'+str(k)+'.txt'
                    else:
                        test_ds = per+'_unlabeled_test_'+str(k)+'.txt'
                     
                    if (args.action != 'latefusion_retest_2models'):    
                        dataloaders = Dataloader(Dataset).test_loader(size, test_ds, batch_size)
                    else:
                        dataloaders = Dataloader(Dataset).test_fusion_loader(size, test_ds, batch_size)
                        
                    
                    class_num, all_imgs = Dataloader(Dataset).count_imgs(test_ds)
                    # load training weights
                    # model_conv.load_state_dict(torch.load('Aug_Adam_'+wts+'_'+Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_'+'k_'+str(k+1)+'.pt'))
                    if args.action == 'retest':
                        # model_conv.load_state_dict(torch.load((str(args.per_teacher)+'_teacher_Adam_'+wts+'_' + Dataset+'_'+ff+'_'+args.network+'_'+per+'_finetune_'
                        #                                         +str(num_student)+'_k_'+str(k+1)+'.pt')))

                        model_conv.load_state_dict(torch.load((str(args.per_teacher)+'_teacher_Adam_'+wts+'_' + Dataset+'_'+ff+'_'+args.network+'_'+per+'_finetune_'
                                        +str(P)+'_'+str(K)+'_No.student_'+str(num_student)+'_k_'+str(k+1)+'.pt')))
                    
                    elif args.action == 'retest_based_both':

                        model_conv.load_state_dict(torch.load((str(args.per_teacher)+'_teacher_Adam_'+wts+'_' + Dataset+'_'+ff+'_'+args.network+'_'+per+'_finetune_based_both_'
                                        +str(P)+'_'+str(K)+'_No.student_'+str(num_student)+'_k_'+str(k+1)+'.pt')))
                    
                    elif args.action == 'latefusion_retest_2models':
                        model_conv_1.load_state_dict(torch.load((str(args.per_teacher)+'_teacher_Adam_'+wts+'_' + Dataset+'_'+ff+'_'+args.network+'_'+per+'_finetune_'
                                       +str(P)+'_'+str(K)+'_No.student_'+str(num_student)+'_k_'+str(k+1)+'.pt')))
                        model_conv_2.load_state_dict(torch.load((str(args.per_teacher)+'_teacher_Adam_'+wts+'_Enh_'+ff+'_'+args.network+'_'+per+'_finetune_'
                                       +str(P)+'_'+str(K)+'_No.student_'+str(num_student)+'_k_'+str(k+1)+'.pt')))
                   
                        
                    # Test accuracy
                    if (args.action != 'latefusion_retest_2models'):
                        k_acc, overall_acc, all_preds, all_labels, all_paths, all_scores = Trainer_models.test_model(model_conv, dataloaders, all_imgs, class_num, args.network)
                    else:
                        k_acc, overall_acc, all_preds, all_labels, all_paths, all_scores = Trainer_models.test_latefusion_model(model_conv_1, model_conv_2, dataloaders, all_imgs, class_num, args.network)
                    
                    single_Dataset_acc.append(k_acc) # 5 k accuracy
                    overall_single_Dataset_acc.append(overall_acc)
                    
                    all_labels_list = all_labels.tolist()
                    all_preds_list = all_preds.tolist()
                    all_scores_list = all_scores.tolist()
                    
                    NN_alltype_all_labels.append(all_labels_list)
                    NN_alltype_all_preds.append(all_preds_list)
                    NN_alltype_all_scores.append(all_scores_list)
                    NN_alltype_all_paths.append(all_paths)
    
                temp=[]
                for iii in single_Dataset_acc:
                    # for each in iii:
                    temp = temp + iii#[each]
                    # single_Dataset_acc_single = temp                                        
                    # Class acc
                    normal_acc_5Ks = temp[slice(0,len(temp),num_classes)]
                    pneumonia_acc_5Ks = temp[slice(1,len(temp),num_classes)]
                    COVID_acc_5Ks = temp[slice(2,len(temp),num_classes)]
                    # Pedicle_acc_5Ks = single_Dataset_acc[slice(3,len(single_Dataset_acc),num_classes)]              
                
                # normal_acc_5Ks.append(normal_acc_Ks)
                # pneumonia_acc_5Ks.append(pneumonia_acc_Ks)
                # COVID_acc_5Ks.append(COVID_acc_Ks)   
                    
                # Calculate mean for each class
                normal_avg_single_Dataset_acc = [round(sum(normal_acc_5Ks)/K_fold,2)]
                pneumonia_avg_single_Dataset_acc = [round(sum(pneumonia_acc_5Ks)/K_fold,2)]
                COVID_avg_single_Dataset_acc = [round(sum(COVID_acc_5Ks)/K_fold,2)]                    
                # Pedicle_avg_single_Dataset_acc = [round(sum(Pedicle_acc_5Ks)/K_fold,2)]
                
                Cross_avg_acc = normal_avg_single_Dataset_acc+pneumonia_avg_single_Dataset_acc+COVID_avg_single_Dataset_acc
                all_avg_single_Dataset_acc.append(Cross_avg_acc)
                
                
                    #                        all_NN_all_Dataset_acc.append(all_Dataset_acc)
                print(args.network+'_'+'Avg'+'_'+'Acc: normal: %.2f, pneumonia: %.2f, COVID: %.2f' % 
                      (normal_avg_single_Dataset_acc[0], pneumonia_avg_single_Dataset_acc[0], COVID_avg_single_Dataset_acc[0]))
                # print(NN+'_'+'Avg'+'_'+'Acc: Pedicle: %.2f, Veterbal_Bone: %.2f' % 
                #       (Pedicle_avg_single_Dataset_acc[0], Veterbal_Bone_avg_single_Dataset_acc[0]))
                print('_' * 10)
                
                #Calcualte var for each class
                normal_var_single_Dataset_acc = [round(np.std(normal_acc_5Ks),2)]
                pneumonia_var_single_Dataset_acc = [round(np.std(pneumonia_acc_5Ks),2)]
                COVID_var_single_Dataset_acc = [round(np.std(COVID_acc_5Ks),2)]                    
                # Pedicle_var_single_Dataset_acc = [round(np.std(Pedicle_acc_5Ks),2)]
               
                Cross_std_acc = normal_var_single_Dataset_acc+pneumonia_var_single_Dataset_acc+COVID_var_single_Dataset_acc
                all_var_single_Dataset_acc.append(Cross_std_acc)
                    
                all_Dataset_acc.append(single_Dataset_acc)
                
                # Calculate overall accuracy
                #Location(2)*finetune(2)*iteration(2)*NN(7)*Datatype(5)*max/avg(1)/All(5 in one list)
                overall_avg_single_Dataset_acc = round(sum(overall_single_Dataset_acc)/K_fold,2)
                overall_all_avg_single_Dataset_acc.append(overall_avg_single_Dataset_acc)
                # Calculate overall variance
                overall_var_single_Dataset_acc = round(np.std(overall_single_Dataset_acc),2)
                overall_all_var_single_Dataset_acc.append(overall_var_single_Dataset_acc)
                # Save all overall accuracy
                overall_all_Dataset_acc.append(overall_single_Dataset_acc)
                # Print result
                print(args.network+'_'+'Average'+'_'+'Acc: %.2f' % (overall_avg_single_Dataset_acc))
                print('_' * 10)
           

        # save variables    
        #import pickle
        file_1 = open(args.action+'_'+wts+'_Adam_'+args.network+'_allsubs_'+Dataset+'_'+args.dataset+'_'+str(args.per_teacher)+'_'+str(P)+'_'+str(K)+'_No.student_'+str(num_student)+'_acc.pickle','wb') #class accurray
        file_2 = open(args.action+'_'+wts+'_Adam_'+args.network+'_allsubs_'+Dataset+'_all_avg_single_'+args.dataset+'_'+str(args.per_teacher)+'_'+str(P)+'_'+str(K)+'_No.student_'+str(num_student)+'_acc.pickle','wb')#
        file_3 = open(args.action+'_'+wts+'_Adam_'+args.network+'_allsubs_'+Dataset+'_all_var_single_'+args.dataset+'_'+str(args.per_teacher)+'_'+str(P)+'_'+str(K)+'_No.student_'+str(num_student)+'_acc.pickle','wb')
        file_4 = open(args.action+'_'+wts+'_Adam_'+args.network+'_allsubs_overall_'+Dataset+'_all_'+args.dataset+'_'+str(args.per_teacher)+'_'+str(P)+'_'+str(K)+'_No.student_'+str(num_student)+'_acc.pickle','wb') # overall accuracy
        file_5 = open(args.action+'_'+wts+'_Adam_'+args.network+'_allsubs_overall_'+Dataset+'_all_avg_single_'+args.dataset+'_'+str(args.per_teacher)+'_'+str(P)+'_'+str(K)+'_No.student_'+str(num_student)+'_acc.pickle','wb')#
        file_6 = open(args.action+'_'+wts+'_Adam_'+args.network+'_allsubs_overall_'+Dataset+'_all_var_single_'+args.dataset+'_'+str(args.per_teacher)+'_'+str(P)+'_'+str(K)+'_No.student_'+str(num_student)+'_acc.pickle','wb')
        #file_3 = open('all_NN_all_Dataset_acc.pickle','wb')
        #pickle.dump(all_max_single_Dataset_acc, file_1)
        pickle.dump(all_Dataset_acc, file_1) # include each class each K
        pickle.dump(all_avg_single_Dataset_acc, file_2) # each class avg over K
        pickle.dump(all_var_single_Dataset_acc, file_3) # each class var over K
        pickle.dump(overall_all_Dataset_acc, file_4) # overall each K
        pickle.dump(overall_all_avg_single_Dataset_acc, file_5) # overall avg over K
        pickle.dump(overall_all_var_single_Dataset_acc, file_6) # overall var over K
        #pickle.dump(all_NN_all_Dataset_acc, file_3)
        
        #save label and preds for metrics
        file_7 = open(args.action+'_'+wts+'_Adam_'+args.network+'_allsubs_'+Dataset+'_alltype_'+args.dataset+'_'+str(args.per_teacher)+'_'+str(P)+'_'+str(K)+'_No.student_'+str(num_student)+'_preds.pickle','wb')#
        file_8 = open(args.action+'_'+wts+'_Adam_'+args.network+'_allsubs_'+Dataset+'_alltype_'+args.dataset+'_'+str(args.per_teacher)+'_'+str(P)+'_'+str(K)+'_No.student_'+str(num_student)+'_label.pickle','wb')
        pickle.dump(NN_alltype_all_preds, file_7) # include each class each K
        pickle.dump(NN_alltype_all_labels, file_8) # each class avg over K
        
        file_9 = open(args.action+'_'+wts+'_Adam_'+args.network+'_allsubs_'+Dataset+'_alltype_'+args.dataset+'_'+str(args.per_teacher)+'_'+str(P)+'_'+str(K)+'_No.student_'+str(num_student)+'_scores.pickle','wb')#
        file_10 = open(args.action+'_'+wts+'_Adam_'+args.network+'_allsubs_'+Dataset+'_alltype_'+args.dataset+'_'+str(args.per_teacher)+'_'+str(P)+'_'+str(K)+'_No.student_'+str(num_student)+'_paths.pickle','wb')
        pickle.dump(NN_alltype_all_scores, file_9) # include each class each K
        pickle.dump(NN_alltype_all_paths, file_10) # each class avg over K
        