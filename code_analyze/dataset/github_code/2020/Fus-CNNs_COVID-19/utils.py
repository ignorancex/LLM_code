#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 21:52:08 2020

@author: endiqq
"""



import torch
import torch.nn as nn
import os
from PIL import Image
import time
import torchvision
from torchvision import transforms
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import numpy as np
import PIL
from barbar import Bar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
             

        
class Xray_Dataset(object):
    def __init__(self, csv_file, dataset, transform=None):
        super(Xray_Dataset, self).__init__()
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform
        self.dataset = dataset
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, idx):   
        cat = self.frame.loc[idx][0].split(' ')
        img_path =  os.path.join(cat[1], self.dataset, cat[2])
        label = int(cat[3])
        if self.dataset == 'CXR_ijcar_mix':
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, img_path

    
class FusionDataset(object):
    def __init__(self, csv_file, dataset_1, dataset_2, transform1 = None, transform2 = None):
        super(FusionDataset, self).__init__()
        
        self.frame = pd.read_csv(csv_file, header = None)
        self.transform1 = transform1
        self.transform2 = transform2
        
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
                  
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, idx):        
        cat = self.frame.loc[idx][0].split(' ')
        
        img_path_1 = os.path.join(cat[1], self.dataset_1, cat[2])
        img_path_2 = os.path.join(cat[1], self.dataset_2, cat[2])
        
        img_path = cat[1] + '/' + cat[2]
        
        label = int(cat[3])
        img_1 = Image.open(img_path_1).convert('RGB')
        img_2 = Image.open(img_path_2)
        
        # if self.transform is not None:
        img_1 = self.transform1(img_1)
        img_2 = self.transform2(img_2)           
        return img_1, img_2, label, img_path
    

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for iidx, content in enumerate(Bar(loader)):
        data, _, _ = content
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    
    return mean, std

def fusion_online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    channels_sum2, channels_squared_sum2, num_batches2 = 0, 0, 0
    
    for iidx, content in enumerate(Bar(loader)):
        data, data2, _, _ = content
        
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
        
        channels_sum2 += torch.mean(data2, dim=[0, 2, 3])
        channels_squared_sum2 += torch.mean(data2**2, dim=[0,2,3])
        num_batches2 += 1
    
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    
    mean2 = channels_sum2/num_batches2
    std2 = (channels_squared_sum2/num_batches2 - mean2**2)**0.5
    
    return mean, std, mean2, std2

class Dataloader:
    def __init__(self, dataset):
        self.Dataset = dataset
 
    def data_loader(self, size, k, batch_size):
                    
        data_transforms = transforms.Compose([transforms.Resize((size, size)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(hue=.05, saturation=.05),
                                              transforms.RandomAffine(10, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                                              transforms.ToTensor()]) 

        dataloaders = {x: torch.utils.data.DataLoader(Xray_Dataset(x+'_ds_'+str(k)+'.txt', self.Dataset, data_transforms), 
                                                      batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}


        all_mean = []
        all_std = []
        for iner in ['train', 'val']:
            print (iner)
            mean, std = online_mean_and_sd(dataloaders[iner])
            print (mean, std)
            all_mean.append(mean.numpy())
            all_std.append(std.numpy())
        
        print (all_mean, all_std)
                        
        data_transforms = {
            'train': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(hue=.05, saturation=.05),
                              transforms.RandomAffine(10, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.ToTensor(),
                              transforms.Normalize(all_mean[0], all_std[0])
                              ]),
            'val': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(hue=.05, saturation=.05),
                              transforms.RandomAffine(10, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.ToTensor(),
                              transforms.Normalize(all_mean[1], all_std[1])
                              ])
            }

        dataloaders = {x: torch.utils.data.DataLoader(Xray_Dataset(x+'_ds_'+str(k)+'.txt', self.Dataset, data_transforms[x]), 
                                      batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}
        
        return dataloaders
    
    def fusion_data_loader(self, size, k, batch_size):
                    
        data_transforms = {
            'train': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                              transforms.RandomAffine(14, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.ToTensor(),
                              ]),
            'val': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                              transforms.RandomAffine(14, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.ToTensor(),
                              ])
            }

        dataloaders = {x: torch.utils.data.DataLoader(FusionDataset(x+'_ds_'+str(k)+'.txt', self.Dataset, 'Enh_ijcar_mix', data_transforms[x], data_transforms[x]), 
                                                      batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}
        
        # dataloaders2 = 
        all_mean = []
        all_std = []
        for iner in ['train', 'val']:
            print (iner)
            mean1, std1, mean2, std2 = fusion_online_mean_and_sd(dataloaders[iner])
            print (mean1, std1, mean2, std2)
            all_mean.append(mean1.numpy())
            all_std.append(std1.numpy())
            all_mean.append(mean2.numpy())
            all_std.append(std2.numpy())
        
        print (all_mean, all_std)
                        
        data_transforms1 = {
            'train': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(hue=.05, saturation=.05),
                              transforms.RandomAffine(14, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.ToTensor(),
                              transforms.Normalize(all_mean[0], all_std[0])
                              ]),
            'val': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(hue=.05, saturation=.05),
                              transforms.RandomAffine(14, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.ToTensor(),
                              transforms.Normalize(all_mean[2], all_std[2])
                              ])
        }
        
        data_transforms2 = {
            'train': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(hue=.05, saturation=.05),
                              transforms.RandomAffine(14, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.ToTensor(),
                              transforms.Normalize(all_mean[1], all_std[1])
                              ]),
            'val': transforms.Compose([transforms.Resize((size, size)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(hue=.05, saturation=.05),
                              transforms.RandomAffine(14, translate = (0.08,0.08), resample=PIL.Image.BILINEAR),
                              transforms.ToTensor(),
                              transforms.Normalize(all_mean[3], all_std[3])
                              ])
        }
        
        dataloaders = {x: torch.utils.data.DataLoader(FusionDataset(x+'_ds_'+str(k)+'.txt', self.Dataset, 'Enh_ijcar_mix', data_transforms1[x], data_transforms2[x]), 
                                                      batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}

        return dataloaders
    
    def test_loader(self, size, test_file, batch_size):
        data_transforms = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])
    
        dataloaders = torch.utils.data.DataLoader(Xray_Dataset(test_file, self.Dataset, data_transforms), batch_size=batch_size, shuffle=True, num_workers=4)
        
        mean, std = online_mean_and_sd(dataloaders)
        print (mean, std)
        mean.numpy(), std.numpy()
        
        data_transforms = transforms.Compose([transforms.Resize((size, size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std),
                                              ])

        dataloaders = torch.utils.data.DataLoader(Xray_Dataset(test_file, self.Dataset, data_transforms), batch_size=batch_size, shuffle=True, num_workers=4)
        
        return dataloaders
    
    def test_fusion_loader(self, size, test_file, batch_size):

        data_transforms = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])  
        dataloaders = torch.utils.data.DataLoader(FusionDataset(test_file, self.Dataset, 'Enh_ijcar_mix', data_transforms, data_transforms), 
                                                  batch_size=batch_size, shuffle=True, num_workers=4)
            
        mean1, std1, mean2, std2 = fusion_online_mean_and_sd(dataloaders)
        print (mean1, std1, mean2, std2)
            
        
        data_transforms1 = transforms.Compose([transforms.Resize((size, size)),
              transforms.ToTensor(),
              transforms.Normalize(mean1, std1)
              ])
        
        data_transforms2 = transforms.Compose([transforms.Resize((size, size)),
              transforms.ToTensor(),
              transforms.Normalize(mean2, std2)
              ])

        dataloaders = torch.utils.data.DataLoader(FusionDataset(test_file, self.Dataset, 'Enh_ijcar_mix', data_transforms1, data_transforms2), 
                                                  batch_size=batch_size, shuffle=True, num_workers=4)
        
        return dataloaders

    
    def count_imgs(self, file):
        class_num = [0, 0, 0]
        frame = pd.read_csv(file, header=None)
        for i in range(len(frame)):
            cat = frame.iloc[i][0].split(' ')
            if cat[3] == '0':
                class_num[0] += 1
            elif cat[3] == '1':
                class_num[1] += 1
            else:
                class_num[-1] += 1
        print (class_num, len(frame))
        return class_num, len(frame)  

class Trainer:
    def __init__(self, lr, num_classes):
        self.ce = nn.CrossEntropyLoss()
        self.lr = lr
        self.num_classes = num_classes
       
    def train_model(self, model, dataloaders, num_epochs, params, writer):
        optimizer = optim.SGD(params, lr=self.lr, momentum=0.9)  
        scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        
        since = time.time()
        val_acc_history = []
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for epoch in range(num_epochs):
                    
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:        
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                    
                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data.
                for index, data in enumerate(Bar(dataloaders[phase])):
                    inputs, labels, img_path = data
                    
                    labels = torch.from_numpy(np.asarray(labels))
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = self.ce(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    if phase == 'train':
                        writer.add_scalar('Avg_batch_Loss/train_loss', loss, epoch * len(dataloaders['train']) + index)
                    else:
                        writer.add_scalar('Avg_batch_Loss/val_loss', loss, epoch * len(dataloaders['val']) + index)
                    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                if phase == 'train':
                    scheduler.step()
                    print (scheduler.get_last_lr()[0])
                    # scheduler.
                    
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                
                if phase == 'train':
                    writer.add_scalar('Epoch_Loss/train_loss', epoch_loss, epoch)
                else:
                    writer.add_scalar('Epoch_Loss/val_loss', epoch_loss, epoch)

                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                
                if phase == 'train':
                    writer.add_scalar('Epoch_Acc/train_acc', epoch_acc, epoch)
                else:
                    writer.add_scalar('Epoch_Acc/val_acc', epoch_acc, epoch)
                
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                    
        last_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
    
        return model, val_acc_history, best_acc, best_model_wts, last_model_wts

    def train_fusion_model(self, model, dataloaders, num_epochs, params, writer):
        optimizer = optim.SGD(params, lr=self.lr, momentum=0.9)  
        scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        
        since = time.time()
        val_acc_history = []
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        all_epochs_loss = []
        for epoch in range(num_epochs):
                    
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
    #            h = model.init_hidden(batch_size)            
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                    
                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data.
                for index, data in enumerate(Bar(dataloaders[phase])):
                    inputs_1, inputs_2, labels, img_path = data
                    
                    # online_mean_and_sd(inputs)             
                    labels = torch.from_numpy(np.asarray(labels))
                    inputs_1 = inputs_1.to(device)
                    inputs_2 = inputs_2.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs_1, inputs_2)
                        loss = self.ce(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    if phase == 'train':
                        writer.add_scalar('Avg_batch_Loss/train_loss', loss, epoch * len(dataloaders['train']) + index)
                    else:
                        writer.add_scalar('Avg_batch_Loss/val_loss', loss, epoch * len(dataloaders['val']) + index)
                    
                    # statistics
                    running_loss += loss.item() * inputs_1.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                # for param_group in optimizer.param_groups:
                #     print(param_group['lr'])
                    
                if phase == 'train':
                    scheduler.step()
                    print (scheduler.get_last_lr()[0])
                    # scheduler.
                    
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                
                if phase == 'train':
                    writer.add_scalar('Epoch_Loss/train_loss', epoch_loss, epoch)
                else:
                    writer.add_scalar('Epoch_Loss/val_loss', epoch_loss, epoch)
                    
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                
                if phase == 'train':
                    writer.add_scalar('Epoch_Acc/train_acc', epoch_acc, epoch)
                else:
                    writer.add_scalar('Epoch_Acc/val_acc', epoch_acc, epoch)
                
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                    
                all_epochs_loss.append(epoch_loss)
                    
        last_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
    
        return model, val_acc_history, best_acc, best_model_wts, last_model_wts#, all_epochs_loss
    
    def test_model(self, model, dataloaders, dataset_sizes, class_num, network):
          
            #Setup model
            model.eval()
            # General accuracy
            running_corrects = 0
            
            # Accuarcy of single class
            normal_running_corrects = 0
            pneumonia_running_corrects = 0
            COVID_running_corrects = 0
        
            count = 0
            # Do test
            for index, data in enumerate(Bar(dataloaders)):
                #get inputs
                img, label, img_path = data        
                img_v = img.to(device)
                label_v = label.to(device)
                # make prediction
                prediction = model(img_v)
                _, preds = torch.max(prediction.data, 1)
                
                # statistics
                running_corrects += torch.sum(preds == label_v)
                # from GPU to CPU
                preds = preds.cpu()
                label = label.cpu()
                
                if index == 0:
                    labell = label
                    predd = preds
                    scores = prediction.data
                    paths = list(img_path)
                    
                else:
                    labell = torch.cat((labell, label), dim=0)
                    predd = torch.cat((predd, preds), dim=0)
                    scores = torch.cat((scores, prediction.data), dim=0)
                    paths = paths+list(img_path)
                   
                # Calculate Class accuracy
                for i in range(len(preds)):
                    if preds.numpy()[i] == label.numpy()[i]:
                        if preds.numpy()[i] == 0:
                            normal_running_corrects = normal_running_corrects+1
                        elif preds.numpy()[i] == 1:
                            pneumonia_running_corrects = pneumonia_running_corrects+1
                        elif preds.numpy()[i] == 2:
                            COVID_running_corrects = COVID_running_corrects+1
                               
                # count how many images
                count = count + 1
                   
            # Overall accuracy                
            overall_acc = (float(running_corrects) / dataset_sizes)*100
            # Each class accuracy
            if class_num[0] != 0:
                normal_acc = [(float(normal_running_corrects) / class_num[0])*100] # Accuracy for single K
            else:
                normal_acc = [0.]
                
            if class_num[1] != 0:
                pneumonia_acc = [(float(pneumonia_running_corrects) / class_num[1])*100]
            else:
                pneumonia_acc = [0.]
                
            if class_num[2] !=0:
                COVID_acc = [(float(COVID_running_corrects) / class_num[2])*100]
            else:
                COVID_acc = [0.]
        
            # Combine each class together
            Class_acc = normal_acc + pneumonia_acc + COVID_acc         
            print(network+'_'+'Single_K_normal_Acc: %.2f, Single_K_pneumonia_Acc: %.2f, Single_K_COVID_acc: %.2f' %
                  (normal_acc[0], pneumonia_acc[0], COVID_acc[0]))
            print(network+'_'+'Overall_acc: %.2f' % (overall_acc))
            print('-' * 30)
            
            return Class_acc, overall_acc, predd, labell, paths, scores
        
    def test_fusion_1model(self, model1, dataloaders, dataset_sizes, class_num, network):

            model1.eval()
            # model2.eval()
            # General accuracy
            running_corrects = 0
            
            # Accuarcy of single class
            normal_running_corrects = 0
            pneumonia_running_corrects = 0
            COVID_running_corrects = 0
            # Pedicle_running_corrects = 0
        
            count = 0
            # Do test
            for index, data in enumerate(Bar(dataloaders)):
                #get inputs
                img_1, img_2, label, img_path = data        
                img_1_v = img_1.to(device)
                img_2_v = img_2.to(device)
                label_v = label.to(device)
                # make prediction
                prediction1 = model1(img_1_v, img_2_v)
                # prediction2 = model2(img_2_v)
        #        prediction = model_conv(img_v) # Used for test, comment when in real
        
                prediction = prediction1            
                _, preds = torch.max(prediction.data, 1)
                
                # statistics
                running_corrects += torch.sum(preds == label_v)
                # from GPU to CPU
                preds = preds.cpu()
                label = label.cpu()
                
                if index == 0:
                    labell = label
                    predd = preds
                    scores = prediction.data
                    paths = list(img_path)
                    # paths_2 = list(img_path_2)
                    
                else:
                    labell = torch.cat((labell, label), dim=0)
                    predd = torch.cat((predd, preds), dim=0)
                    scores = torch.cat((scores, prediction.data), dim=0)
                    paths = paths+list(img_path)
   
                # Calculate Class accuracy
                for i in range(len(preds)):
                    if preds.numpy()[i] == label.numpy()[i]:
                        if preds.numpy()[i] == 0:
                            normal_running_corrects = normal_running_corrects+1
                        elif preds.numpy()[i] == 1:
                            pneumonia_running_corrects = pneumonia_running_corrects+1
                        elif preds.numpy()[i] == 2:
                            COVID_running_corrects = COVID_running_corrects+1
                               
                # count how many images
                count = count + 1
                   
            # Overall accuracy                
            overall_acc = (float(running_corrects) / dataset_sizes)*100
            # Each class accuracy
            normal_acc = [(float(normal_running_corrects) / class_num[0])*100] # Accuracy for single K    
            pneumonia_acc = [(float(pneumonia_running_corrects) / class_num[1])*100]
            COVID_acc = [(float(COVID_running_corrects) / class_num[2])*100]
        
            # Combine each class together
            Class_acc = normal_acc + pneumonia_acc + COVID_acc         
            print(network+'_'+'Single_K_normal_Acc: %.2f, Single_K_pneumonia_Acc: %.2f, Single_K_COVID_acc: %.2f' %
                  (normal_acc[0], pneumonia_acc[0], COVID_acc[0]))
            print(network+'_'+'Overall_acc: %.2f' % (overall_acc))
            print('-' * 30)
            
            return Class_acc, overall_acc, predd, labell, paths, scores