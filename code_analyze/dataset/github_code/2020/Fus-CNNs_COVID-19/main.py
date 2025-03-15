#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 21:51:54 2020

@author: endiqq
"""


import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from transfer_models import Transfer_learning as tf_learning
from utils import Xray_Dataset, Dataloader, Trainer, FusionDataset
import pickle
import numpy as np
from Fusion_models import Late_Fusion_Net as latefusion
from Fusion_models import Mid_Fusion_Net as middlefusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train') # train_unlabel
parser.add_argument('--network', default="Res50")
parser.add_argument('--method', default='Conv')
parser.add_argument('--dataset', default = 'CXR_ijcar_mix') 

args = parser.parse_args()


def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
    
    # hyper-parameters
    K_fold = 5
    nepoch = 50
    num_classes = 3
    batch_size = 32
    lr = 1e-3

    finetune = ['Entire'] #finetune entire model
    wts = 'Best'

    # define model    
    Trainer_models = Trainer(lr, num_classes)
       
    K_Accuracy=[] #best validation accuracy
    His_Accuracy=[]#all validation accuracy from all epoechs    
    # train models
    if (args.action == 'train' or args.action == 'latefusion' or args.action == 'middlefusion'):
        for ff in finetune:
            print (ff)
            print('-' * 30)   
            for iidx, Dataset in enumerate([args.dataset]):
                print (Dataset)
                print('-' * 30)      
                for k in range(K_fold):            
                    print ('K_fold = %d' % k)
                    
                    if args.action == 'train':
                        # tensorboard
                        writer = SummaryWriter('runs/'+ args.network+'_'+Dataset +'_'+ str(k))                     
                        model, size, pretrained, num_ftrs = tf_learning().def_model(args.network, num_classes, ff, use_pretrained = True)
                        model_conv = model.to(device)
                        
                        dataloaders = Dataloader(Dataset).data_loader(size, k, batch_size)
                        
                        params_to_update = []                    
                        print("Params to learn:")
                        for name,param in model_conv.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                print("\t",name)
                                
                        _, hist, best_acc, Best_model_wts, Last_model_wts = Trainer_models.train_model(model_conv, dataloaders, nepoch, params_to_update, writer)                              
                        # save the last validation model weights and the best validation model weights
                        torch.save(Best_model_wts, 'Aug_Best_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt')#save weights
                        torch.save(Last_model_wts, 'Aug_Last_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt')
                        K_Accuracy.append(best_acc)
                        His_Accuracy.append(hist)
                    
                    elif args.action == 'latefusion' or args.action == 'middlefusion':
                        writer = SummaryWriter('runs/'+ args.action+'_'+args.network+'_'+ Dataset +'_'+ str(k))                     
                        
                        model1, size, pretrained, num_ftrs = tf_learning().def_model(args.network, num_classes, ff, use_pretrained = True)
                        model2, size, pretrained, num_ftrs = tf_learning().def_model(args.network, num_classes, ff, use_pretrained = True)
                        
                        model1.load_state_dict(torch.load('Aug_Best_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))
                        # my_model1 = pre_model_conv1#.features                                 
                        model2.load_state_dict(torch.load('Aug_Best_Enh_ijcar_mix_'+ff+'_'+args.network+'_Enh_ijcar_mix_k_'+str(k+1)+'.pt'))                    
                        
                        if args.action == 'latefusion':
                            model = latefusion(args.network, model1, model2, num_classes)
                        else:
                            embedding_dim = num_ftrs
                            model = middlefusion(args.network, model1, model2, embedding_dim, num_classes, args.method)
                        
                        model_conv = model.to(device)
                        
                        dataloaders = Dataloader(Dataset).fusion_data_loader(size, k, batch_size)
     
                        params_to_update = []                    
                        print("Params to learn:")
                        for name,param in model_conv.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                print("\t",name)
                                
                        _, hist, best_acc, Best_model_wts, Last_model_wts = Trainer_models.train_fusion_model(model_conv, dataloaders, nepoch, params_to_update, writer)
                        
                        if args.action == 'latefusion':
                            torch.save(Best_model_wts, ('CNN_'+args.action+'_Best_lastfc_'+str(num_classes)+'class_'+args.network+'_Sum_'
                                                        +Dataset+'_Enh_ijcar_mix_k_'+str(k+1)+'.pt'))#save weights
                            torch.save(Last_model_wts, ('CNN_'+args.action+'_Last_lastfc_'+str(num_classes)+'class_'+args.network+'_Sum_'
                                                        +Dataset+'_Enh_ijcar_mix_k_'+str(k+1)+'.pt'))                                        
                        else:
                            torch.save(Best_model_wts, ('CNN_'+args.action+'_Best_lastconv_'+str(num_classes)+'class_'+args.network+'_'+args.method+'_'
                                                        +Dataset+'_Enh_ijcar_mix_k_'+str(k+1)+'.pt'))#save weights
                            torch.save(Last_model_wts, ('CNN_'+args.action+'_Last_lastconv_'+str(num_classes)+'class_'+args.network+'_'+args.method+'_'
                                                        +Dataset+'_Enh_ijcar_mix_k_'+str(k+1)+'.pt'))
                            
                        K_Accuracy.append(best_acc)
                        His_Accuracy.append(hist)
                                  
        # Convert to number from tensor
        K_Accuracy_num = []
        for i in range(len(K_Accuracy)):
            K_Accuracy_num.append(round(float(K_Accuracy[i].cpu()),4)*100)    
        His_Accuracy_num = []
        for i in range(len(His_Accuracy)):
            one_NN = []
            for j in range(len(His_Accuracy[i])):
                one_NN.append(round(float(His_Accuracy[i][j].cpu()),4)*100)
            His_Accuracy_num.append(one_NN)
                            
        #Save validation accuracy
        file_1 = open(args.network+'_'+args.action+'_'+Dataset+'_best_acc.pickle','wb')
        file_2 = open(args.network+'_'+args.action+'_'+Dataset+'_Validation_all_acc.pickle','wb')
        pickle.dump(K_Accuracy_num, file_1)
        pickle.dump(His_Accuracy_num, file_2)
        
    # Test model
    if  (args.action == 'test' or args.action == 'test_latefusion' or args.action == 'test_middlefusion'):

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
            
            for iidx, Dataset in enumerate([args.dataset]):
                print (Dataset)
                print('-' * 30)
                
                if args.action == 'test':
                    model_conv, size, pretrained, num_ftrs = tf_learning().def_model(args.network, num_classes, feature_extract = 'fc', use_pretrained=False)
                    embedding_dim = num_ftrs
                    model_conv = model_conv.to(device)
                    
                else:
                    model_conv_1, size, pretrained, num_ftrs = tf_learning().def_model(args.network, num_classes, feature_extract = 'fc', use_pretrained=False)
                    model_conv_2, size, pretrained, num_ftrs = tf_learning().def_model(args.network, num_classes, feature_extract = 'fc', use_pretrained=False)
                    embedding_dim = num_ftrs
                    
                    if args.action == 'test_latefusion':
                        model = latefusion(args.network, model_conv_1, model_conv_2, num_classes)
                    else:
                        model = middlefusion(args.network, model_conv_1, model_conv_2, embedding_dim, num_classes, args.method)
                        
                    model_conv = model.to(device)
                                    
                single_Dataset_acc = []
                overall_single_Dataset_acc = []
                
                for k in range(K_fold):
                    print (k)
                        
                    test_ds = 'test_ds_'+str(k)+'.txt'
                        
                    if args.action == 'test':    
                        dataloaders = Dataloader(Dataset).test_loader(size, test_ds, batch_size)
                    else:
                        dataloaders = Dataloader(Dataset).test_fusion_loader(size, test_ds, batch_size)
                                            
                    class_num, all_imgs = Dataloader(Dataset).count_imgs(test_ds)
                    
                    # load training weights
                    if args.action == 'test':
                        model_conv.load_state_dict(torch.load('Aug_'+ wts+ '_' + Dataset+'_'+ff+'_'+args.network+'_'+Dataset+'_k_'+str(k+1)+'.pt'))             
                    elif args.action == 'test_latefusion':    
                        model_conv.load_state_dict(torch.load(('CNN_'+args.action.split('_')[-1]+'_'+wts+'_lastfc_'+str(num_classes)+'class_'+args.network+'_Sum_'
                                                        +Dataset+'_Enh_ijcar_mix_k_'+str(k+1)+'.pt')))
                    elif args.action == 'test_middlefusion':
                        model_conv.load_state_dict(torch.load(('CNN_'+args.action.split('_')[-1]+'_'+wts+'_lastconv_'+str(num_classes)+'class_'+args.network+'_'+args.method+'_'
                                                        +Dataset+'_Enh_ijcar_mix_k_'+str(k+1)+'.pt')))                    
                        
                    # Test accuracy
                    if args.action == 'test':
                        k_acc, overall_acc, all_preds, all_labels, all_paths, all_scores = Trainer_models.test_model(model_conv, dataloaders, all_imgs, class_num, args.network)
                    else:
                        k_acc, overall_acc, all_preds, all_labels, all_paths, all_scores = Trainer_models.test_fusion_1model(model_conv, dataloaders, all_imgs, class_num, args.network)

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
                    temp = temp + iii#[each]
                    normal_acc_5Ks = temp[slice(0,len(temp),num_classes)]
                    pneumonia_acc_5Ks = temp[slice(1,len(temp),num_classes)]
                    COVID_acc_5Ks = temp[slice(2,len(temp),num_classes)] 
                    
                # Calculate mean for each class
                normal_avg_single_Dataset_acc = [round(sum(normal_acc_5Ks)/K_fold,2)]
                pneumonia_avg_single_Dataset_acc = [round(sum(pneumonia_acc_5Ks)/K_fold,2)]
                COVID_avg_single_Dataset_acc = [round(sum(COVID_acc_5Ks)/K_fold,2)]                    
                
                Cross_avg_acc = normal_avg_single_Dataset_acc+pneumonia_avg_single_Dataset_acc+COVID_avg_single_Dataset_acc
                all_avg_single_Dataset_acc.append(Cross_avg_acc)
                
                print(args.network+'_'+'Avg'+'_'+'Acc: normal: %.2f, pneumonia: %.2f, COVID: %.2f' % 
                      (normal_avg_single_Dataset_acc[0], pneumonia_avg_single_Dataset_acc[0], COVID_avg_single_Dataset_acc[0]))
                print('_' * 10)
                
                #Calcualte var for each class
                normal_var_single_Dataset_acc = [round(np.std(normal_acc_5Ks),2)]
                pneumonia_var_single_Dataset_acc = [round(np.std(pneumonia_acc_5Ks),2)]
                COVID_var_single_Dataset_acc = [round(np.std(COVID_acc_5Ks),2)]                    
               
                Cross_std_acc = normal_var_single_Dataset_acc+pneumonia_var_single_Dataset_acc+COVID_var_single_Dataset_acc
                all_var_single_Dataset_acc.append(Cross_std_acc)
                    
                all_Dataset_acc.append(single_Dataset_acc)
                
                # Calculate overall accuracy
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
        
        # save accuracy                                   
        if args.action == 'test':
            # save variables    
            #import pickle
            file_1 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_acc.pickle','wb') #class accurray
            file_2 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_all_avg_single_acc.pickle','wb')#
            file_3 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_all_var_single_acc.pickle','wb')
            file_4 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_acc.pickle','wb') # overall accuracy
            file_5 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_avg_single_acc.pickle','wb')#
            file_6 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_var_single_acc.pickle','wb')

            pickle.dump(all_Dataset_acc, file_1) # each class acc each K
            pickle.dump(all_avg_single_Dataset_acc, file_2) # each class avg acc over K
            pickle.dump(all_var_single_Dataset_acc, file_3) # each class var over K
            pickle.dump(overall_all_Dataset_acc, file_4) # overall each K
            pickle.dump(overall_all_avg_single_Dataset_acc, file_5) # overall avg over K
            pickle.dump(overall_all_var_single_Dataset_acc, file_6) # overall var over K
            
            #save label and preds for metrics
            file_7 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_preds.pickle','wb')#
            file_8 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_label.pickle','wb')
            pickle.dump(NN_alltype_all_preds, file_7) # preds each K
            pickle.dump(NN_alltype_all_labels, file_8) # label each K
            
            file_9 = open(args.action+'_'+wts+'_Adam_'+args.network+'_allsubs_'+Dataset+'_alltype_scores.pickle','wb')#
            file_10 = open(args.action+'_'+wts+'_Adam_'+args.network+'_allsubs_'+Dataset+'_alltype_paths.pickle','wb')
            pickle.dump(NN_alltype_all_scores, file_9) # raw prediction score each K
            pickle.dump(NN_alltype_all_paths, file_10) # image path each K

        elif args.action=='test_middlefusion':
            file_1 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_Enh_ijcar_mix_acc.pickle','wb') #class accurray
            file_2 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_all_avg_single_Enh_ijcar_mix_acc.pickle','wb')#
            file_3 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_all_var_single_Enh_ijcar_mix_acc.pickle','wb')
            file_4 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_Enh_ijcar_mix_acc.pickle','wb') # overall accuracy
            file_5 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_avg_single_Enh_ijcar_mix_acc.pickle','wb')#
            file_6 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_var_single_Enh_ijcar_mix_acc.pickle','wb')

            pickle.dump(all_Dataset_acc, file_1) # each class acc each K
            pickle.dump(all_avg_single_Dataset_acc, file_2) # each class avg over K
            pickle.dump(all_var_single_Dataset_acc, file_3) # each class var over K
            pickle.dump(overall_all_Dataset_acc, file_4) # overall each K
            pickle.dump(overall_all_avg_single_Dataset_acc, file_5) # overall avg over K
            pickle.dump(overall_all_var_single_Dataset_acc, file_6) # overall var over K
            #pickle.dump(all_NN_all_Dataset_acc, file_3)
            
            #save label and preds for metrics
            file_7 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_preds.pickle','wb')#
            file_8 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_label.pickle','wb')
            pickle.dump(NN_alltype_all_preds, file_7) # preds each K
            pickle.dump(NN_alltype_all_labels, file_8) # label each K
            
            file_9 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_scores.pickle','wb')#
            file_10 = open(args.action+'_'+args.method+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_paths.pickle','wb')
            pickle.dump(NN_alltype_all_scores, file_9) # raw predication score each K
            pickle.dump(NN_alltype_all_paths, file_10) # image path each K
            
        else:
            file_1 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_Enh_ijcar_mix_acc.pickle','wb') #class accurray
            file_2 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_all_avg_single_Enh_ijcar_mix_acc.pickle','wb')#
            file_3 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_all_var_single_Enh_ijcar_mix_acc.pickle','wb')
            file_4 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_Enh_ijcar_mix_acc.pickle','wb') # overall accuracy
            file_5 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_avg_single_Enh_ijcar_mix_acc.pickle','wb')#
            file_6 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_overall_'+Dataset+'_all_var_single_Enh_ijcar_mix_acc.pickle','wb')

            pickle.dump(all_Dataset_acc, file_1) # each class acc each K
            pickle.dump(all_avg_single_Dataset_acc, file_2) # each class avg over K
            pickle.dump(all_var_single_Dataset_acc, file_3) # each class var over K
            pickle.dump(overall_all_Dataset_acc, file_4) # overall each K
            pickle.dump(overall_all_avg_single_Dataset_acc, file_5) # overall avg over K
            pickle.dump(overall_all_var_single_Dataset_acc, file_6) # overall var over K
            #pickle.dump(all_NN_all_Dataset_acc, file_3)
            
            #save label and preds for metrics
            file_7 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_preds.pickle','wb')#
            file_8 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_label.pickle','wb')
            pickle.dump(NN_alltype_all_preds, file_7) # preds each K
            pickle.dump(NN_alltype_all_labels, file_8) # label each K
            
            file_9 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_scores.pickle','wb')#
            file_10 = open(args.action+'_'+wts+'_'+args.network+'_allsubs_'+Dataset+'_alltype_Enh_ijcar_mix_paths.pickle','wb')
            pickle.dump(NN_alltype_all_scores, file_9) # raw predication score each K
            pickle.dump(NN_alltype_all_paths, file_10) # image path each K
            
