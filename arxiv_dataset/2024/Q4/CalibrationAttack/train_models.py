import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
from sklearn.utils import shuffle

from tqdm import tqdm, trange
import io

import argparse
import random

import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.optim import lr_scheduler
import collections
from torch import Tensor
import torch.nn as nn

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import time
from tqdm import tqdm, trange
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.models as torchmodels  


from transformers import ViTFeatureExtractor, ViTForImageClassification, AutoImageProcessor, SwinForImageClassification

import transformers

from models import MlpMixer

from get_data import get_dataset, get_attack_dataset

from utils import ECE, get_top_results, calculate_ks_error, one_hot_encode, ReliabilityDiagram, WarmupCosineSchedule , dense_to_onehot, ece_score, softmax

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from recalibration import TempScalingModel, SplinesModel, MDTempScalingModel


from netcal.regularization import DCAPenalty, MMCEPenalty

from sam import SAM

from models import ModelPT

from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD

import math 

import random


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
    
    
def cuda_transfer(images, target):
  images = images.cuda(non_blocking=True)
  target = target.type(torch.LongTensor).cuda(non_blocking=True)
  return images, target

                
def get_model(nclasses=101, model_type="resnet"):
  ngpus = torch.cuda.device_count()
  nchannels=3
  if model_type=="resnet":
    model = torchmodels.resnet50(pretrained=True)
    num_features = model.fc.in_features
    print(num_features)
    model.fc = nn.Linear(num_features, nclasses)
  if model_type=="resnet18":
    model = torchmodels.resnet18(pretrained=True)
    num_features = model.fc.in_features
    print(num_features)
    model.fc = nn.Linear(num_features, nclasses)
  if model_type=="resnet152":
    model = torchmodels.resnet152(pretrained=True)
    num_features = model.fc.in_features
    print(num_features)
    model.fc = nn.Linear(num_features, nclasses)    
  elif model_type=="vit":
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", revision ="1ba429d32753f33a0660b80ac6f43a3c80c18938")
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, nclasses)
    #print(dict(model.named_modules()))
  elif model_type=="vitL":
    model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224-in21k")
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, nclasses)
  elif model_type=="SwinTiny":
    model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, nclasses)
  elif model_type=="SwinBase":
    model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, nclasses)

  print ('Model succesfully loaded')    
  model = model.cuda()
  return model
  
# finetune basic model  
def finetune_model(dataset_name, n_classes, batch_size, epochs, model_type="resnet", evaluate='accuracy', variation="a", imbalance=False, imbalance_ratio=0.01):

    tr_set = get_dataset(dataset_name,'train', imbalance=imbalance, imbalance_ratio=imbalance_ratio)
    
    val_set = get_dataset(dataset_name,'val')   
    
    train_loader = torch.utils.data.DataLoader(tr_set, batch_size=batch_size,\
                              shuffle=True, num_workers=0, pin_memory=True)  
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size*2,
                      shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()

    N = len(tr_set)
    iterations = (N//batch_size+1)*epochs
    print(iterations)
    model = get_model(nclasses=n_classes, model_type=model_type)
    
    if model_type == "resnet" or model_type == "resnet18" or model_type == "resnet152":
        optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, iterations)
    elif model_type == "vit"  or model_type == "vitL" or model_type == "SwinTiny" or model_type == "SwinBase":
        optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, iterations)


    best_result = 0.0
    current_result = 0.0
    

    for _ in trange(epochs, desc="Epoch"):    
        model.train()    
        for i, (images, target) in enumerate(train_loader):
            images, target = cuda_transfer(images, target)

            if model_type == "vit" or model_type == "vitL" or model_type == "SwinTiny" or model_type == "SwinBase":
                output = model(images).logits
            else:    
                output = model(images)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            if model_type == "vit" or model_type == "vitL" or model_type == "SwinTiny" or model_type == "SwinBase":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        labels_list = []
        logits_list = []
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(test_loader):
              images, target = cuda_transfer(images, target)
              if model_type == "vit" or model_type == "vitL" or model_type == "SwinTiny" or model_type == "SwinBase":
                output = model(images).logits
              else:    
                output = model(images)
              logits_list.append(output.cpu())
              labels_list.append(target.cpu())
              
        labels_list = torch.cat(labels_list)
        logits_list = torch.cat(logits_list)
        
        softmaxes_list = F.softmax(logits_list, dim=1)
        _, predictions_list = torch.max(softmaxes_list, dim=1)
    
        accuracy = accuracy_score(labels_list.detach().cpu().numpy(),predictions_list.detach().cpu().numpy())
        macf1 = f1_score(labels_list.detach().cpu().numpy(),predictions_list.detach().cpu().numpy(), average='macro')
        
        current_result = accuracy if evaluate=="accuracy" else macf1
        
        if current_result >= best_result: 
            best_result = current_result
            print("best model " + evaluate + ": " + str(current_result))
            if imbalance==False:
                model_string = "./models/trained_models/" + dataset_name + "_" + model_type + "_" + variation + ".pt"
            else:
                model_string = "./models/trained_models/" + dataset_name + "_" + model_type + "_" + variation + "_imbalanced_" + str(int(1/imbalance_ratio)) + ".pt"
            torch.save(model.state_dict(), model_string)
    
    
def train_DCA(dataset_name, n_classes, batch_size, epochs, model_type="resnet", evaluate='accuracy', weight=10, sigma=0.2, variation="a"):

    tr_set = get_dataset(dataset_name,'train')
    
    val_set = get_dataset(dataset_name,'val')   
    
    train_loader = torch.utils.data.DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)  
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size*2, shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()
    
    dca = DCAPenalty(weight=weight)
    
    N = len(tr_set)
    iterations = (N//batch_size+1)*epochs
    print(iterations)
    model = get_model(nclasses=n_classes, model_type=model_type)
    
    if model_type == "resnet":
        optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, iterations)
    elif model_type == "vit":
        optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, iterations)


    best_result = 0.0
    current_result = 0.0

    if model_type == "vit":
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k", do_resize=False, do_normalize=False)  

    for _ in trange(epochs, desc="Epoch"):    
        model.train()    
        for i, (images, target) in enumerate(train_loader):
            noisy_images = torch.clone(images)
            counter = 0
            for j, image in enumerate(images):
                noisy_images[j] = image + torch.randn(image.size()) * sigma     

            bb = [t.numpy() for t in noisy_images]
            if model_type == "vit":
                images = feature_extractor(bb, return_tensors="pt")
                images = images["pixel_values"]
                images, target = cuda_transfer(images, target)
            else:
                images, target = cuda_transfer(noisy_images, target)

            if model_type == "vit":
                output = model(images).logits
            else:    
                output = model(images)
            
            normal_loss = criterion(output, target)
            
            dca_loss = dca(output, target)
            
            loss = normal_loss + dca_loss
            
            optimizer.zero_grad()
            loss.backward()
            if model_type == "vit":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        labels_list = []
        logits_list = []
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(test_loader):
              bb = [t.numpy() for t in images]
              if model_type == "vit":
                images = feature_extractor(bb, return_tensors="pt")
                images = images["pixel_values"]
              images, target = cuda_transfer(images, target)
              if model_type == "vit":
                output = model(images).logits
              else:    
                output = model(images)
              logits_list.append(output.cpu())
              labels_list.append(target.cpu())
              
        labels_list = torch.cat(labels_list)
        logits_list = torch.cat(logits_list)
        
        softmaxes_list = F.softmax(logits_list, dim=1)
        _, predictions_list = torch.max(softmaxes_list, dim=1)
    
        accuracy = accuracy_score(labels_list.detach().cpu().numpy(),predictions_list.detach().cpu().numpy())
        macf1 = f1_score(labels_list.detach().cpu().numpy(),predictions_list.detach().cpu().numpy(), average='macro')
        
        current_result = accuracy if evaluate=="accuracy" else macf1
        
        if current_result >= best_result: 
            best_result = current_result
            print("best model " + evaluate + ": " + str(current_result))
            model_string = "./models/trained_models/defense/" + dataset_name + "_" + model_type +  "_" + variation +  "_DCA.pt"
            torch.save(model.state_dict(), model_string)
    return model


def train_SAM(dataset_name, n_classes, batch_size, epochs, model_type="resnet", evaluate='accuracy', sigma=0.2, variation="a"):

    tr_set = get_dataset(dataset_name,'train')
    
    val_set = get_dataset(dataset_name,'val')   
    
    train_loader = torch.utils.data.DataLoader(tr_set, batch_size=batch_size,shuffle=True, num_workers=0, pin_memory=True)  
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size*2,shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()

    N = len(tr_set)
    iterations = (N//batch_size+1)*epochs
    print(iterations)
    model = get_model(nclasses=n_classes, model_type=model_type)
    
    if model_type == "resnet":
        lr = 0.01
        wd = 5e-4
    elif model_type == "vit":
        lr = 0.01
        wd = 5e-4

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=lr, weight_decay=wd, momentum=0.9)
    
    if model_type == "resnet" or model_type == 'vit':    
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, iterations)   
        
    best_result = 0.0
    current_result = 0.0


    if model_type == "vit":
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k", do_resize=False, do_normalize=False)  


    for _ in trange(epochs, desc="Epoch"):    
        model.train()    
        for i, (images, target) in enumerate(train_loader):
            noisy_images = torch.clone(images)
            for j, image in enumerate(images):
                noisy_images[j] = image + torch.randn(image.size()) * sigma     

            bb = [t.numpy() for t in noisy_images]
            if model_type == "vit":
                images = feature_extractor(bb, return_tensors="pt")
                images = images["pixel_values"]
                images, target = cuda_transfer(images, target)
            else:
                images, target = cuda_transfer(noisy_images, target)

            enable_running_stats(model)
            if model_type == "vit":
                output = model(images).logits
            else:    
                output = model(images)

            loss = criterion(output, target) 

            loss.backward()

            optimizer.first_step(zero_grad=True)
            disable_running_stats(model)
            
            if model_type == "vit":
                output = model(images).logits
            else:    
                output = model(images)            
            
            criterion(output, target).backward()  
                                    
            if model_type == "vit":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2) 
                              
            optimizer.second_step(zero_grad=True)
            scheduler.step()
                
        labels_list = []
        logits_list = []
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(test_loader):
              bb = [t.numpy() for t in images]
              if model_type == "vit":
                images = feature_extractor(bb, return_tensors="pt")
                images = images["pixel_values"]
              images, target = cuda_transfer(images, target)
              if model_type == "vit":
                output = model(images).logits
              else:    
                output = model(images)
              logits_list.append(output.cpu())
              labels_list.append(target.cpu())
              
        labels_list = torch.cat(labels_list)
        logits_list = torch.cat(logits_list)
        
        softmaxes_list = F.softmax(logits_list, dim=1)
        _, predictions_list = torch.max(softmaxes_list, dim=1)
    
        accuracy = accuracy_score(labels_list.detach().cpu().numpy(),predictions_list.detach().cpu().numpy())
        macf1 = f1_score(labels_list.detach().cpu().numpy(),predictions_list.detach().cpu().numpy(), average='macro')
        
        current_result = accuracy if evaluate=="accuracy" else macf1
        
        if current_result >= best_result: 
            best_result = current_result
            print("best model " + evaluate + ": " + str(current_result))
            model_string = "./models/trained_models/defense/" + dataset_name + "_" + model_type +  "_" + variation + "_SAM.pt"
            torch.save(model.state_dict(), model_string)

    return model


class ViTAdversarial(nn.Module):
    def __init__(self, model):
        super(ViTAdversarial, self).__init__()

        self.model = model
    def forward(self, x):
        return self.model(x).logits


def adversarial_training(dataset_name, n_classes, batch_size, epochs, model_type="resnet", evaluate='accuracy', sigma=0.0, variation="a", attack_steps=1):

    tr_set = get_dataset(dataset_name,'train')
    
    val_set = get_dataset(dataset_name,'val')   
    
    train_loader = torch.utils.data.DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)  
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size*2, shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()
    

    N = len(tr_set)
    iterations = (N//batch_size+1)*epochs
    print(iterations)
    model = get_model(nclasses=n_classes, model_type=model_type)

    if model_type == "vit":
        model = ViTAdversarial(model)

    if model_type == "resnet":
        optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
        #optimizer = torch.optim.SGD(model.parameters(), 0.005, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, iterations)
    elif model_type == "vit":
        optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, iterations)

    # apply the attack
    attack = LinfPGD(steps=attack_steps)

    best_result = 0.0
    current_result = 0.0

    if model_type == "vit":
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k", do_resize=False, do_normalize=False)  

    min_value = 1000
    max_value = -1000
    for i, (images, target) in enumerate(train_loader):
        curr_max = torch.max(images).item()
        curr_min =  torch.min(images).item() 
        if curr_min < min_value:
            min_value = curr_min
        if curr_max > max_value:
            max_value = curr_max         


    for _ in trange(epochs, desc="Epoch"):    
        model.train()    
        for i, (images, target) in enumerate(train_loader):

            images, target = cuda_transfer(images, target)

            model = model.eval()
            fmodel = PyTorchModel(model, bounds=(min_value - 0.01, max_value + 0.01), device='cuda:0')
            fmodel = fmodel.transform_bounds((min_value - 0.01, max_value + 0.01)) 

            raw_advs, noisy_images, success = attack(fmodel, images, target, epsilons=0.1)
            model.train()   
            bb = [t.numpy() for t in noisy_images.cpu()]
            if model_type == "vit":
                images = feature_extractor(bb, return_tensors="pt")
                images = images["pixel_values"]
                images, target = cuda_transfer(images, target)
            else:
                images, target = cuda_transfer(noisy_images, target)

            if model_type == "vit_b":
                output = model(images).logits
            else:    
                output = model(images)
            
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            if model_type == "vit":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        labels_list = []
        logits_list = []
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(test_loader):
              bb = [t.numpy() for t in images]
              if model_type == "vit":
                images = feature_extractor(bb, return_tensors="pt")
                images = images["pixel_values"]
              images, target = cuda_transfer(images, target)
              if model_type == "vit_b":
                output = model(images).logits
              else:    
                output = model(images)
              logits_list.append(output.cpu())
              labels_list.append(target.cpu())
              
        labels_list = torch.cat(labels_list)
        logits_list = torch.cat(logits_list)
        
        softmaxes_list = F.softmax(logits_list, dim=1)
        _, predictions_list = torch.max(softmaxes_list, dim=1)
    
        accuracy = accuracy_score(labels_list.detach().cpu().numpy(),predictions_list.detach().cpu().numpy())
        macf1 = f1_score(labels_list.detach().cpu().numpy(),predictions_list.detach().cpu().numpy(), average='macro')
        
        current_result = accuracy if evaluate=="accuracy" else macf1
        
        if current_result >= best_result: 
            best_result = current_result
            print("best model " + evaluate + ": " + str(current_result))
            model_string = "./models/trained_models/defense/" + dataset_name + "_" + model_type +  "_" + variation +  "_AT.pt"
            torch.save(model.model.state_dict(), model_string)

    return model.model


def PGD_Calibration_Attack(model, x, y, eps, n_iters, attack_type, alpha=2/255, model_type='resnet'):

    mean = np.reshape([0.485, 0.456, 0.406], [1, 3, 1, 1])
    std = np.reshape([0.229, 0.224, 0.225], [1, 3, 1, 1])
    
    mean, std = torch.from_numpy(mean.astype(np.float32)).cuda(), torch.from_numpy(std.astype(np.float32)).cuda()

    ori_images = x.clone()
    images = x
    labels = y
    loss = nn.CrossEntropyLoss()
    m = torch.nn.Softmax(dim=-1)


    vit_extractor = None
    if model_type == "vit":
        vit_extractor =  ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k", do_resize=False, do_normalize=False)
  
    with torch.no_grad():

        if model_type=="vit":
            temp_output = model(vit_extractor((images - mean) / std, return_tensors="pt")["pixel_values"].cuda(non_blocking=True)).logits
        else:
            temp_output = model((images - mean) / std)

    _, clean_predictions = torch.max(F.softmax(temp_output, dim=1), dim=1)

    for i in range(n_iters) :    
      
        if model_type=="vit":
            bb = [((t - mean) / std).cpu().numpy() for t in images]
            temp_images = vit_extractor(bb, return_tensors="pt")["pixel_values"].cuda(non_blocking=True)
            temp_images.requires_grad = True
            outputs = model(temp_images.squeeze()).logits
        else:
            images.requires_grad = True
            outputs = model((images - mean) / std)
        model.zero_grad()

        cost = loss(m(outputs), labels).cuda()
        cost.backward()

        if attack_type == 'underconf':
            if model_type=="vit":
                adv_images = images + alpha*temp_images.grad.sign().squeeze()
            else:
                adv_images = images + alpha*images.grad.sign()
        elif attack_type == 'overconf':
            if model_type=="vit":
                adv_images = images - alpha*temp_images.grad.sign().squeeze()
            else:
                adv_images = images - alpha*images.grad.sign()

        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)

        if attack_type == 'underconf':
            eta =  F.dropout(eta, p=0.95)
        elif attack_type == 'overconf':
            eta =  F.dropout(eta, p=0.1)

        images_temp = torch.clamp(ori_images + eta, min=0, max=1).detach_()

        with torch.no_grad():
            if model_type=="vit":
                bb = [((t - mean) / std).cpu().numpy() for t in images_temp]
                temp_images = vit_extractor(bb, return_tensors="pt")["pixel_values"].cuda(non_blocking=True)
                outputs2 = model(temp_images.squeeze()).logits

            else:
                outputs2 = model((images_temp - mean) / std)

        softmaxes_ece = F.softmax(outputs2, dim=1)
        confidences_ece, predictions_ece = torch.max(softmaxes_ece, dim=1)

        label_unchanged = torch.eq(predictions_ece , clean_predictions)

        if model_type=="vit":
            temp_images = temp_images.detach_()

        images = images.detach_()

        images[label_unchanged] = images_temp[label_unchanged]

    return images


def adversarial_training_calibration_attack(dataset_name, n_classes, batch_size, epochs, model_type="resnet", evaluate='accuracy', variation="a", attack_steps=1):

    tr_set = get_dataset(dataset_name,'train')
    
    val_set = get_dataset(dataset_name,'val')   
    
    train_loader = torch.utils.data.DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)  
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size*2, shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()
    

    N = len(tr_set)
    iterations = (N//batch_size+1)*epochs
    print(iterations)
    model = get_model(nclasses=n_classes, model_type=model_type)



    if model_type == "vit":
        model = ViTAdversarial(model)

    if model_type == "resnet":
        optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
        #optimizer = torch.optim.SGD(model.parameters(), 0.005, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, iterations)
    elif model_type == "vit":
        optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, iterations)

    attack = PGD_Calibration_Attack

    best_result = 0.0
    current_result = 0.0


    for epoch in trange(epochs, desc="Epoch"):    
        model.train()    
        for i, (images, target) in enumerate(train_loader):

            images, target = cuda_transfer(images, target)

            model = model.eval()

            noisy_images_under = attack(model, images, target, 0.1, attack_steps, "underconf", model_type=model_type) 

            noisy_images_over = attack(model, images, target, 0.1, attack_steps, "overconf", model_type=model_type) 

            model.train()   

            if model_type == "vit_b":
                output = model(images).logits
            else:    

                output = model(noisy_images_under)
            
            loss_1 = criterion(output, target)

            if model_type == "vit_b":
                output = model(images).logits
            else:    

                output = model(noisy_images_over)

            loss_2 = criterion(output, target)

            loss = loss_1 + loss_2

            optimizer.zero_grad()
            loss.backward()
            if model_type == "vit":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        labels_list = []
        logits_list = []
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(test_loader):
              images, target = cuda_transfer(images, target)
              if model_type == "vit_b":
                output = model(images).logits
              else:    
                output = model(images)
              logits_list.append(output.cpu())
              labels_list.append(target.cpu())
              
        labels_list = torch.cat(labels_list)
        logits_list = torch.cat(logits_list)
        
        softmaxes_list = F.softmax(logits_list, dim=1)
        _, predictions_list = torch.max(softmaxes_list, dim=1)
    
        accuracy = accuracy_score(labels_list.detach().cpu().numpy(),predictions_list.detach().cpu().numpy())
        macf1 = f1_score(labels_list.detach().cpu().numpy(),predictions_list.detach().cpu().numpy(), average='macro')
        
        current_result = accuracy if evaluate=="accuracy" else macf1
        
        if current_result >= best_result: 
            best_result = current_result
            print("best model " + evaluate + ": " + str(current_result))
            model_string = "./models/trained_models/defense/" + dataset_name + "_" + model_type +  "_" + variation +  "_AT_calibration_attk.pt"
            torch.save(model.state_dict(), model_string)

    return model



# evaluate a trained model on test set
def evaluate_model(model, dataset_name, model_type="non_vit"):

    test_set = get_dataset(dataset_name,'test')   
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128,
                      shuffle=False, num_workers=0, pin_memory=True)

    labels_list = []
    logits_list = []
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
          images, target = cuda_transfer(images, target)

          if model_type == "vit":
            output = model(images).logits
          else:    
            output = model(images)

          logits_list.append(output.cpu())
          labels_list.append(target.cpu())
          
    labels_ece = torch.cat(labels_list)
    logits_ece = torch.cat(logits_list)
    ece_criterion = ECE().to("cpu")
    overall_ece = ece_criterion(logits_ece, labels_ece).item()
    
    softmaxes_ece = F.softmax(logits_ece, dim=1)
    confidences_ece, predictions_ece = torch.max(softmaxes_ece, dim=1)

    accuracy = accuracy_score(labels_ece.detach().cpu().numpy(),predictions_ece.detach().cpu().numpy()).item()
    macf1 = f1_score(labels_ece.detach().cpu().numpy(),predictions_ece.detach().cpu().numpy(), average='macro').item()

    avg_predicted_conf = torch.mean(confidences_ece).item() 
    
    n=-1
    y_probs = softmaxes_ece.detach().cpu().numpy()
    y_labels = labels_ece.detach().cpu().numpy()

    ks_error = calculate_ks_error(n, y_probs, y_labels, confidences_ece).item()
    
    
    print("Evaluation Results")
    print('Accuracy: {:.4f}, F1: {:.4f}, Average Confidence: {:.4f}, ECE: {:.4f}, Brier Score: {:.4f},KS Error: {:.4f}'.format(accuracy, macf1, avg_predicted_conf, overall_ece, ks_error))



# generate calibration stats            
def evaluate_calibration(model, x, y, adv="og"):
 
    labels_list = []
    logits_list = []

    labels_ece = torch.from_numpy(y)
    logits = model.predict(x)
    logits_ece = torch.from_numpy(logits)
    
    ece_criterion = ECE().to("cpu")
    overall_ece = ece_criterion(logits_ece, labels_ece).item()

    rd = ReliabilityDiagram().to("cpu")
    conf_list, acc_list = rd(logits_ece, labels_ece)
    
    softmaxes_ece = F.softmax(logits_ece, dim=1)
    confidences_ece, predictions_ece = torch.max(softmaxes_ece, dim=1)


    bin_boundaries = torch.linspace(0, 1, 15 + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidence_bins = []
    x_axis = []


    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = confidences_ece.gt(bin_lower.item()) * confidences_ece.le(bin_upper.item())
                prop_in_bin = in_bin.float().sum()
                confidence_bins.append(prop_in_bin.item())
                x_axis.append("{:.2f}".format(bin_lower.item()) + "-" + "{:.2f}".format(bin_upper.item()))
    

    fig, axs = plt.subplots(1, 1, figsize=(12, 10))

    axs.bar(np.arange(len(confidence_bins)),confidence_bins, width=1.0, edgecolor='k', linewidth=1)
    axs.set_title('Confidence Score Distribution', fontsize=16)
    axs.set_xticks(np.arange(len(confidence_bins)))
    axs.set_xticklabels(x_axis, fontsize=7)
    axs.set_ylabel('Confidence', fontsize=14)

    fig.savefig(adv + '_confidence_distribution.pdf')

    accuracy = accuracy_score(labels_ece.detach().cpu().numpy(),predictions_ece.detach().cpu().numpy()).item()
    macf1 = f1_score(labels_ece.detach().cpu().numpy(),predictions_ece.detach().cpu().numpy(), average='macro').item()

    avg_predicted_conf = torch.mean(confidences_ece).item() 
    
    n=-1
    y_probs = softmaxes_ece.detach().cpu().numpy()
    y_labels = labels_ece.detach().cpu().numpy()

    ks_error = calculate_ks_error(n, y_probs, y_labels, confidences_ece).item()
    
    
    brier_score = 0.0
    return accuracy, macf1, avg_predicted_conf, overall_ece, brier_score, ks_error, logits_ece, conf_list, acc_list


# train temperature scaling
def create_temp_scale_model(model, dataset_name, batch_size, model_type='resnet'):

    val_set = get_dataset(dataset_name,'val')   
    
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size*2,shuffle=False, num_workers=0, pin_memory=True)

    tempScaleModel = TempScalingModel(model,model_type)
    tempScaleModel.cuda()
    tempScaleModel.set_temperature(val_loader)

    return tempScaleModel, tempScaleModel.temperature
    

def create_splines_model(model, dataset_name, batch_size):

    val_set = get_dataset(dataset_name,'val')   
    
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size*2,shuffle=False, num_workers=0, pin_memory=True)
    
    calibratedModel = SplinesModel(model)
    calibratedModel.cuda()
    calibratedModel.set_splines(val_loader)

    return calibratedModel


class LogitCompression(nn.Module):

    def __init__(self, model, model_type='resnet'):
        super(LogitCompression, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.2)

        if model_type == 'vit':
            self.vit_on = True
        else:
            self.vit_on = False

    def forward(self, input):
        if self.vit_on:
            logits = self.model(input).logits
        else:
            logits = self.model(input)
        return self.logit_scale(logits)

    def logit_scale(self, logits):
        softmaxes_ece = F.softmax(logits, dim=1)
        confidences_ece, _ = torch.max(softmaxes_ece, dim=1)
        bin_boundaries = torch.linspace(0, 1, 3 + 1)  
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        new_bin_uppers = [0.86666666666,0.93333333333, 1.0]
        new_bin_lowers = [0.8,0.86666666666,0.93333333333]
        target_val = 0.2666666667

        new_logits_list = []
        for i in range(confidences_ece.shape[0]):
            
            bin_idx = torch.logical_and((bin_uppers > confidences_ece[i].item()), (bin_lowers <= confidences_ece[i].item()))
            bin_idx = torch.where(bin_idx == True)[0].item()
            min_val = new_bin_lowers[bin_idx]
            max_val  = new_bin_uppers[bin_idx]

            new_conf_score = min_val + (1/15)*((confidences_ece[i].item() - bin_lowers[bin_idx])/target_val)

            found_temp = False
            for j in range(800):
                new_temperature = nn.Parameter(torch.ones(1) * (1.0 - 0.00125*j))

                new_temperature = new_temperature.unsqueeze(1).expand(1, logits.size(1)).cuda()
                temp_logits = logits[i] / new_temperature 
                temp_softmaxes = F.softmax(temp_logits, dim=1)
                temp_confidence, _ = torch.max(temp_softmaxes, dim=1)
                if abs(temp_confidence.item() - new_conf_score) <= 0.01:
                    new_logits_list.append(temp_logits.squeeze())
                    found_temp = True
                    break
            if found_temp == False:
                new_logits_list.append(logits[i].squeeze())

        new_logits_list  = torch.stack(new_logits_list)

        return new_logits_list.squeeze()



def create_compressed_model(model, model_type='resnet'):


    tempScaleModel = LogitCompression(model, model_type)
    tempScaleModel.cuda()

    return tempScaleModel



device = torch.device('cuda:0')
verbose = False

def loss(y, logits, targeted=False, loss_type='margin_loss'):
    if loss_type == 'margin_loss':
        preds_correct_class = (logits * y).sum(1, keepdims=True)
        diff = preds_correct_class - logits
        diff[y] = np.inf
        margin = diff.min(1, keepdims=True)
        loss = margin * -1 if targeted else margin
    elif loss_type == 'cross_entropy':
        probs = softmax(logits)
        loss = -np.log(probs[y])
        loss = loss * -1 if not targeted else loss
    else:
        raise ValueError('Wrong loss.')
    return loss.flatten()


def predict(x, model, batch_size, device, mean=[0], std=[1]):
    if isinstance(x, np.ndarray):
        x = np.floor(x * 255.0) / 255.0
        x = ((x - np.array(mean)[np.newaxis, :, np.newaxis, np.newaxis]) / np.array(std)[np.newaxis, :, np.newaxis, np.newaxis]).astype(np.float32)
        n_batches = math.ceil(x.shape[0] / batch_size)
        logits_list = []
        with torch.no_grad():
            for counter in range(n_batches):
                if verbose: print('predicting', counter, '/', n_batches, end='\r')
                x_curr = torch.as_tensor(x[counter * batch_size:(counter + 1) * batch_size], device=device)
                logits_list.append(model(x_curr).detach().cpu().numpy())
        logits = np.vstack(logits_list)
        return logits
    else:
        return model(x)


class AAALinear(nn.Module):
    def __init__(self, model, dataset, arch, 
        device=device, batch_size=1000, attractor_interval=6, reverse_step=1, num_iter=100, calibration_loss_weight=5, optimizer_lr=0.1, do_softmax=False, **kwargs):
        super(AAALinear, self).__init__()
        self.dataset = dataset

        self.model_type = arch

        self.cnn = model.to(device).eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.loss = loss
        self.batch_size = batch_size
        self.device = device

        self.attractor_interval = attractor_interval
        self.reverse_step = reverse_step
        self.dev = 0.5
        self.optimizer_lr = optimizer_lr
        self.calibration_loss_weight = calibration_loss_weight
        self.num_iter = num_iter
        self.arch_ori = arch
        self.arch = '%s_AAAlinear-Lr-%.1f-Ai-%d-Cw-%d' % (self.arch_ori, self.reverse_step, self.attractor_interval, self.calibration_loss_weight)
        self.temperature = 1 # 2.08333 #
        self.do_softmax = do_softmax

    def set_hp(self, reverse_step, attractor_interval=6, calibration_loss_weight=5):
        self.attractor_interval = attractor_interval
        self.reverse_step = reverse_step
        self.calibration_loss_weight = calibration_loss_weight
        self.arch = '%s_AAAlinear-Lr-%.1f-Ai-%d-Cw-%d' % (self.arch_ori, self.reverse_step, self.attractor_interval, self.calibration_loss_weight)

    def forward_undefended(self, x): return predict(x, self.cnn, self.batch_size, self.device, self.mean, self.std)
    
    def get_tuned_temperature(self):
        t_dict = {
            'Standard': 2.08333,
            'resnet50': 1.1236,
            'resnext101_32x8d': 1.26582,
            'vit_b_16': 0.94,
            'wide_resnet50_2': 1.20482,
            'Rebuffi2021Fixing_28_10_cutmix_ddpm': 0.607,
            'Salman2020Do_50_2': 0.83,
            'Dai2021Parameterizing': 0.431,
            'Rade2021Helper_extra': 0.58
        }
        return t_dict.get(self.arch_ori, None)

    def temperature_rescaling(self, x_val, y_val, step_size=0.001):
        ts, eces = [], []
        ece_best, y_best = 100, 1
        y_pred = self.forward_undefended(x_val)
        for t in np.arange(0, 1, step_size):
            y_pred1 = y_pred / t
            y_pred2 = y_pred * t

            ts += [t, 1/t]
            ece1, ece2 = ece_score(y_pred1, y_val), ece_score(y_pred2, y_val)
            eces += [ece1, ece2]
            if ece1 < ece_best: 
                ece_best = ece1
                t_best = t
            if ece2 < ece_best: 
                ece_best = ece2
                t_best = 1/t
            print('t-curr=%.3f, acc=%.2f, %.2f, ece=%.4f, %.4f, t-best=%.5f, ece-best=%.4f' % 
            (t, (y_pred1.argmax(1) == y_val.argmax(1)).mean() * 100, (y_pred2.argmax(1) == y_val.argmax(1)).mean() * 100, 
            ece1 * 100, ece2 * 100,
            t_best, ece_best * 100))
        self.temperature = t_best

        plt.rcParams["figure.dpi"] = 500
        plt.rcParams["font.family"] = "times new roman"
        plt.scatter(ts, eces, color='#9467bd')
        plt.xscale('log')
        plt.xlabel('temperature')
        plt.ylabel('ece on validation set')
        plt.savefig('demo/t-%s-%.4f.png' % (self.arch, self.temperature))
        plt.close()

    def temperature_rescaling_with_aaa(self, x_val, y_val, step_size=0.001):
        print("hi")
        self.temperature = self.get_tuned_temperature()
        if self.temperature is not None: return

        ts, eces = [], []
        ece_best, y_best = 100, 1
        for t in np.arange(0, 1, step_size):
            self.temperature = t
            y_pred1 = self.forward(x_val)
            self.temperature = 1/t
            y_pred2 = self.forward(x_val)

            ts += [t, 1/t]
            ece1, ece2 = ece_score(y_pred1, y_val), ece_score(y_pred2, y_val)
            eces += [ece1, ece2]
            if ece1 < ece_best: 
                ece_best = ece1
                t_best = t
            if ece2 < ece_best: 
                ece_best = ece2
                t_best = 1/t
            print('t-curr=%.3f, acc=%.2f, %.2f, ece=%.4f, %.4f, t-best=%.5f, ece-best=%.4f' % 
            (t, (y_pred1.argmax(1) == y_val.argmax(1)).mean() * 100, (y_pred2.argmax(1) == y_val.argmax(1)).mean() * 100, 
            ece1 * 100, ece2 * 100,
            t_best, ece_best * 100))
        self.temperature = t_best

        plt.rcParams["figure.dpi"] = 500
        plt.rcParams["font.family"] = "times new roman"
        plt.scatter(ts, eces, color='#9467bd')
        plt.xscale('log')
        plt.xlabel('temperature')
        plt.ylabel('ece on validation set')
        plt.savefig('demo/taaa-%s-%.4f.png' % (self.arch, self.temperature))
        plt.close()

    def forward(self, x):
        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []

        for counter in range(n_batches):
            with torch.no_grad():
                if verbose: print('predicting', counter, '/', n_batches, end='\r')
                x_curr = x[counter * self.batch_size:(counter + 1) * self.batch_size]
                if isinstance(x, np.ndarray): x_curr = torch.as_tensor(x_curr, device=self.device) 
                if self.model_type == "vit_b_16":
                    logits = self.cnn(x_curr).logits
                else:    
                    logits = self.cnn(x_curr)                
            
            logits_ori = logits.detach()
            prob_ori = F.softmax(logits_ori / self.temperature, dim=1)
            prob_max_ori = prob_ori.max(1)[0] ###
            value, index_ori = torch.topk(logits_ori, k=2, dim=1)
            mask_first = torch.zeros(logits.shape, device=self.device)
            mask_first[torch.arange(logits.shape[0]), index_ori[:, 0]] = 1
            mask_second = torch.zeros(logits.shape, device=self.device)
            mask_second[torch.arange(logits.shape[0]), index_ori[:, 1]] = 1
            
            margin_ori = value[:, 0] - value[:, 1]
            attractor = ((margin_ori / self.attractor_interval + self.dev).round() - self.dev) * self.attractor_interval
            target = attractor - self.reverse_step * (margin_ori - attractor)
            diff_ori = (margin_ori - target)
            real_diff_ori = margin_ori - attractor
            with torch.enable_grad():
                logits.requires_grad = True
                optimizer = torch.optim.Adam([logits], lr=self.optimizer_lr)
                i = 0 
                los_reverse_rate = 0
                prd_maintain_rate = 0
                for i in range(self.num_iter):
                    prob = F.softmax(logits, dim=1)
                    loss_calibration = ((prob * mask_first).max(1)[0] - prob_max_ori).abs().mean() # better
                    value, index = torch.topk(logits, k=2, dim=1) 
                    margin = value[:, 0] - value[:, 1]

                    diff = (margin - target)
                    real_diff = margin - attractor
                    loss_defense = diff.abs().mean()
                    
                    loss = loss_defense + loss_calibration * self.calibration_loss_weight
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    los_reverse_rate = ((real_diff * real_diff_ori) < 0).float().mean()
                    prd_maintain_rate = (index_ori[:, 0] == index[:, 0]).float().mean()
                logits_list.append(logits.detach().cpu())

        logits = torch.vstack(logits_list)
        if isinstance(x, np.ndarray): logits = logits.numpy()
        if self.do_softmax: logits = softmax(logits)
        return logits


def create_AAA_model(model, dataset_name, batch_size, model_type="resnet"):
    
    val_set = get_dataset(dataset_name,'val')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size*2,shuffle=False, num_workers=0, pin_memory=True)   
    val_data, val_targets = [], []
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            val_data.append(images)
            val_targets.append(target)
            
    val_data = torch.cat(val_data)
    val_targets = torch.cat(val_targets)

    if model_type=="resnet":
        arch = 'resnet50'
    if model_type=="vit":
        arch = 'vit_b_16'        
    calibratedModel = AAALinear(model, dataset_name, arch, device=device, batch_size=batch_size, attractor_interval=4, reverse_step=1, num_iter=100, calibration_loss_weight=5, optimizer_lr=0.1, do_softmax=False)
    calibratedModel.temperature_rescaling_with_aaa(val_data, val_targets)

    return calibratedModel



def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p


def pseudo_gaussian_pert_rectangles(x, y):
    delta = np.zeros([x, y])
    x_c, y_c = x // 2 + 1, y // 2 + 1

    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
              max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

        counter2[0] -= 1
        counter2[1] -= 1

    delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def meta_pseudo_gaussian_pert(s):
    delta = np.zeros([s, s])
    n_subsquares = 2
    if n_subsquares == 2:
        delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
        delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        if np.random.rand(1) > 0.5: delta = np.transpose(delta)

    elif n_subsquares == 4:
        delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
        delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def calibration_attack_temp_scale(model, x, y, eps, n_iters, p_init, attack_type, y_test, max_margin=None):
    """ The Linf calibration attack """
    np.random.seed(0) 
    targeted = False
    
    min_val, max_val = 0, 1 if x.max() <= 1 else 255
    c, h, w = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]

    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
    
    x_best = np.clip(x + 0*init_delta, min_val, max_val) # no itialization
    
    
    if attack_type=='overconf':
        loss_type='margin_loss_overconf'
    
    elif attack_type=='underconf':
        loss_type='margin_loss_overconf'
        
    logits = model.predict(x_best)
    
    initial_logits = torch.from_numpy(logits)
    _, initial_predictions = torch.max(F.softmax(initial_logits, dim=1), dim=1)
    initial_predictions = initial_predictions.numpy()
    
    
    loss_min = model.loss(y, logits, targeted, loss_type=loss_type)
    margin_min = model.loss(y, logits, targeted, loss_type=loss_type)
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query
    
    if max_margin is not None:
        goal_margin = np.copy(margin_min)
        for i, margin in np.ndenumerate(margin_min):
            if attack_type=='underconf':
                goal_margin[i] = max_margin - 0.09
            elif attack_type=='overconf':
                goal_margin[i] = max_margin + 0.09
    
    complete_attacks = 0


    time_start = time.time()
    metrics = np.zeros([n_iters, 7])
    for i_iter in range(n_iters - 1):
        if max_margin is not None:
            if attack_type=='underconf':
                idx_to_fool = margin_min > goal_margin
            elif attack_type=='overconf':
                idx_to_fool = margin_min < goal_margin
        
        else:
            if attack_type=='underconf':
                idx_to_fool = margin_min > 0.1
            elif attack_type=='overconf':
                idx_to_fool = margin_min < 0.99
            
        x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
        loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
        deltas = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-eps, eps], size=[c, 1, 1])
  
        x_new = np.clip(x_curr + deltas, min_val, max_val)

        logits = model.predict(x_new)
        loss = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        class_loss = model.loss(y_curr, logits, targeted, loss_type='margin_loss')

        
        if attack_type=='underconf':
            new_logits = torch.from_numpy(logits)
            _, new_predictions = torch.max(F.softmax(new_logits, dim=1), dim=1)
            new_predictions = new_predictions.numpy()
            
            idx_improved = loss < loss_min_curr
            class_unchanged = class_loss > 0.0
           
            idx_improved = np.logical_and(idx_improved, class_unchanged)
            
            same_predictions = initial_predictions[idx_to_fool] == new_predictions
            idx_improved = np.logical_and(idx_improved, same_predictions)
            
        elif attack_type=='overconf':

            new_logits = torch.from_numpy(logits)
            _, new_predictions = torch.max(F.softmax(new_logits, dim=1), dim=1)
            new_predictions = new_predictions.numpy()

            idx_improved = loss > loss_min_curr
            class_unchanged = class_loss > 0.0
            idx_improved = np.logical_and(idx_improved, class_unchanged)
            
            same_predictions = initial_predictions[idx_to_fool] == new_predictions
            idx_improved = np.logical_and(idx_improved, same_predictions)
            #'''

        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        acc = (margin_min > 0.0).sum() / n_ex_total


        if max_margin is not None:
            if attack_type=='underconf':
                complete_attacks = (margin_min > goal_margin).mean()
            elif attack_type=='overconf':
                complete_attacks = (margin_min < goal_margin).mean()        
        
        else: 
            if attack_type=='underconf':
                complete_attacks = (margin_min > 0.1).mean()
            elif attack_type=='overconf':
                complete_attacks = (margin_min < 0.99).mean()
        
        acc_corr = (margin_min > 0.0).mean()
 
        if max_margin is not None:
            if attack_type=='underconf':
                mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= goal_margin]), np.median(n_queries[margin_min <= goal_margin])
            elif attack_type=='overconf':
                mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min >= goal_margin]), np.median(n_queries[margin_min >= goal_margin])
        else:
            if attack_type=='underconf':
                mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0.1]), np.median(n_queries[margin_min <= 0.1])
            elif attack_type=='overconf':
                mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min >= 0.99]), np.median(n_queries[margin_min >= 0.99])
            
        avg_margin_min = np.mean(margin_min)
        time_total = time.time() - time_start
            
        metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq_ae, margin_min.mean(), time_total]


        if complete_attacks == 0:
            break

    print("Number of complete attacks: " + str(complete_attacks))



    return x_best
    


def calculate_fourier_spectrum(im, typ='MFS'):
    im = im.unsqueeze(0).numpy()

    fft = np.fft.fft2(im)
    if typ == 'MFS':
        fourier_spectrum = np.abs(fft)
    elif typ == 'PFS':
        fourier_spectrum = np.abs(np.angle(fft))

    return fourier_spectrum


def calculate_spectra(images, typ='MFS'):
    fs = []   
    for i in range(images.shape[0]):
        image = images[i,:]
        fourier_image = calculate_fourier_spectrum(image, typ=typ)
        fs.append(fourier_image.flatten())
    return fs



import utils
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.linear_model import LinearRegression


def create_md_temp_scale_model(model, model_type, dataset_name, batch_size, n_cls, use_spectral=False):

    model = model.cuda()
    x_test, y_test = get_attack_dataset(dataset_name, n_ex=50, split='val')

    y_target = y_test
    
    y_target_onehot = utils.dense_to_onehot(y_target, n_cls=n_cls)

    model_attack = ModelPT(model, batch_size, model_type=model_type)
    x_adv_under = calibration_attack_temp_scale(model_attack, x_test, y_target_onehot, 0.05, 20, 0.05, 'underconf', y_target)


    x_adv_over = calibration_attack_temp_scale(model_attack, x_test, y_target_onehot, 0.05, 20, 0.05, 'overconf', y_target)


    ds_x = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    og_loader = DataLoader(ds_x, batch_size=batch_size, shuffle=False, num_workers=0)


    tempScaleModel = TempScalingModel(model,model_type)
    tempScaleModel.cuda()
    tempScaleModel.set_temperature(og_loader)

    og_temp = tempScaleModel.temperature
    print(og_temp)


    ds_x = torch.utils.data.TensorDataset(torch.tensor(x_adv_under, dtype=torch.float), torch.from_numpy(y_test))

    underconf_loader = DataLoader(ds_x, batch_size=batch_size, shuffle=False, num_workers=0)


    tempScaleModel.set_temperature(underconf_loader)
    under_temp = tempScaleModel.temperature       

    print(under_temp)


    ds_x = torch.utils.data.TensorDataset(torch.tensor(x_adv_over, dtype=torch.float), torch.from_numpy(y_test))

    overconf_loader = DataLoader(ds_x, batch_size=batch_size, shuffle=False, num_workers=0)


    tempScaleModel.set_temperature(overconf_loader)
    over_temp = tempScaleModel.temperature       

    print(over_temp)


    if model_type == 'resnet':
        return_nodes = {'flatten': 'flatten'}
        feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)


        labels_list = []
        features_list = []
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(og_loader):
                images, target = cuda_transfer(images, target)
                output = feature_extractor(images)
                if use_spectral == True:
                    fourier_maps = torch.tensor(calculate_spectra(output["flatten"].cpu()))
                    features_list.append(fourier_maps)
                else:
                    features_list.append(output["flatten"].cpu())
                labels_list.append(og_temp.cpu().unsqueeze(1).expand(output["flatten"].size(0), 1))
                
        labels_list = torch.cat(labels_list)
        features_list = torch.cat(features_list)

        labels_list_under = []
        features_list_under = []
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(underconf_loader):
                images, target = cuda_transfer(images, target)  
                output = feature_extractor(images)

                if use_spectral == True:
                    fourier_maps = torch.tensor(calculate_spectra(output["flatten"].cpu()))
                    features_list_under.append(fourier_maps)
                else:
                    features_list_under.append(output["flatten"].cpu())

                labels_list_under.append(under_temp.cpu().unsqueeze(1).expand(output["flatten"].size(0), 1))
                
        labels_list_under = torch.cat(labels_list_under)
        features_list_under = torch.cat(features_list_under)


        labels_list_over = []
        features_list_over = []
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(overconf_loader):
                images, target = cuda_transfer(images, target)  
                output = feature_extractor(images)

                if use_spectral == True:
                    fourier_maps = torch.tensor(calculate_spectra(output["flatten"].cpu()))
                    features_list_over.append(fourier_maps)
                else:
                    features_list_over.append(output["flatten"].cpu())


                labels_list_over.append(over_temp.cpu().unsqueeze(1).expand(output["flatten"].size(0), 1))
                
        labels_list_over = torch.cat(labels_list_over)
        features_list_over = torch.cat(features_list_over)



        features_list = torch.cat((features_list,features_list_under, features_list_over), 0).numpy()
        labels_list = torch.cat((labels_list,labels_list_under, labels_list_over), 0).numpy()

        temp_find_function = LinearRegression().fit(features_list, labels_list)


        newtempScaleModel = MDTempScalingModel(model,temp_find_function)

    elif model_type == 'vit':
        labels_list = []
        features_list = []
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(og_loader):
                images, target = cuda_transfer(images, target)
                if model_type == "vit":
                    output = model(images, output_hidden_states=True).hidden_states[-1].view(images.shape[0], -1)
                    
                if use_spectral == True:
                    fourier_maps = torch.tensor(calculate_spectra(output.cpu()))
                    features_list.append(fourier_maps)
                else:                    
                    features_list.append(output.cpu())
                labels_list.append(og_temp.cpu().unsqueeze(1).expand(output.size(0), 1))
                
        labels_list = torch.cat(labels_list)
        features_list = torch.cat(features_list)

        labels_list_under = []
        features_list_under = []
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(underconf_loader):
                images, target = cuda_transfer(images, target)
                if model_type == "vit":
                    output = model(images, output_hidden_states=True).hidden_states[-1].view(images.shape[0], -1)

                if use_spectral == True:
                    fourier_maps = torch.tensor(calculate_spectra(output.cpu()))
                    features_list_under.append(fourier_maps)
                else:
                    features_list_under.append(output.cpu())
                labels_list_under.append(under_temp.cpu().unsqueeze(1).expand(output.size(0), 1))
                
        labels_list_under = torch.cat(labels_list_under)
        features_list_under = torch.cat(features_list_under)


        labels_list_over = []
        features_list_over = []
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(overconf_loader):
                images, target = cuda_transfer(images, target)  
                output = model(images, output_hidden_states=True).hidden_states[-1].view(images.shape[0], -1)


                if use_spectral == True:
                    fourier_maps = torch.tensor(calculate_spectra(output.cpu()))
                    features_list_over.append(fourier_maps)
                else:
                    features_list_over.append(output.cpu())
                labels_list_over.append(over_temp.cpu().unsqueeze(1).expand(output.size(0), 1))
                
        labels_list_over = torch.cat(labels_list_over)
        features_list_over = torch.cat(features_list_over)

        features_list = torch.cat((features_list,features_list_under, features_list_over), 0).numpy()
        labels_list = torch.cat((labels_list,labels_list_under, labels_list_over), 0).numpy()

        temp_find_function = LinearRegression().fit(features_list, labels_list)

        newtempScaleModel = MDTempScalingModel(model,temp_find_function,model_type)


    return newtempScaleModel

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define parameters.')
    parser.add_argument('--model_type', type=str, default='resnet', help='Model name.') # resnet or mlp_mixer
    parser.add_argument('--dataset', type=str, default='caltech', help='dataset name.')
    parser.add_argument('--train', type=str, default="True", help='whether to train or not (ie evaluate existing model).')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for model.')
    parser.add_argument('--epochs', type=int, default=6, help='Batch size for model.')
    parser.add_argument('--eval_metric', type=str, default='accuracy', help='Evaluation metric for best performance.')
    parser.add_argument('--defense', type=str, default='None', choices=['None', 'label_smoothing', 'DCA', 'SAM', 'MMCE', 'Spline', 'AT', 'AAA', 'CAAT','CTS', 'MDTS'], help='Calibration defense method to train.')
    parser.add_argument('--noise', type=float, default=0.1, help='Amount of gaussian noise to apply for adverserial defense(valid for label smoothing, dca, SAM)')
    parser.add_argument('--version', type=str, default='a', help='which of three models to test on')
    parser.add_argument('--imbalance', type=str, default='False', help='whether to train imbalanced model')
    parser.add_argument('--ratio', type=float, default=0.01, help='imnbalance ratio')


    print(transformers.__version__)
    args = parser.parse_args()
    if not (args.model_type=='resnet' or args.model_type=='resnet18' or args.model_type=='resnet152' or args.model_type=='mlp_mixer' or args.model_type=='vit' or args.model_type == "vitL" or args.model_type == "SwinTiny" or args.model_type == "SwinBase" ):
        print("ERROR: improper model type")
        sys.exit()

    if args.dataset == "caltech":
        n_classes = 101
    elif args.dataset == "cifar100" :
        n_classes = 100   
    elif args.dataset == "gtsrb":
        n_classes = 43   


    if args.defense =="None":
        if args.train=="True": 
            if args.imbalance=='False':  
                finetune_model(args.dataset, n_classes, args.batch_size, args.epochs, model_type=args.model_type, evaluate=args.eval_metric,variation=args.version)
            elif args.imbalance=='True' and args.dataset == "cifar100":
                finetune_model(args.dataset, n_classes, args.batch_size, args.epochs, model_type=args.model_type, evaluate=args.eval_metric,variation=args.version, imbalance=True, imbalance_ratio=args.ratio)
        else:
            model = get_model(nclasses=n_classes, model_type=args.model_type)
            if args.imbalance=='False': 
                saved_model_name = "./models/trained_models/" + args.dataset + "_" + args.model_type + "_" + args.version +".pt"
                model.load_state_dict(torch.load(saved_model_name))
            else:
                saved_model_name = "./models/trained_models/" + args.dataset + "_" + args.model_type + "_" + args.version +"_imbalanced_" + str(int(1/args.ratio)) + ".pt"
                model.load_state_dict(torch.load(saved_model_name))                
            evaluate_model(model, args.dataset, model_type=args.model_type)
    
    else:
        if args.defense == 'DCA':
            model = train_DCA(args.dataset, n_classes, args.batch_size, args.epochs, model_type=args.model_type, evaluate=args.eval_metric, weight=10, sigma=args.noise, variation=args.version)
            evaluate_model(model, args.dataset, model_type=args.model_type)
        elif args.defense == 'AT':
            model = adversarial_training(args.dataset, n_classes, args.batch_size, args.epochs, model_type=args.model_type, evaluate=args.eval_metric, variation=args.version, attack_steps=4)
            evaluate_model(model, args.dataset, model_type=args.model_type) 
        elif args.defense == 'CAAT':
            model = adversarial_training_calibration_attack(args.dataset, n_classes, args.batch_size, args.epochs, model_type=args.model_type, evaluate=args.eval_metric, variation=args.version, attack_steps=10)
            evaluate_model(model, args.dataset, model_type=args.model_type)    
        elif args.defense == 'SAM':
            model = train_SAM(args.dataset, n_classes, args.batch_size, args.epochs, model_type=args.model_type, evaluate=args.eval_metric, sigma=args.noise, variation=args.version)
            evaluate_model(model, args.dataset, model_type=args.model_type)
        elif args.defense == 'AAA':
            model = get_model(nclasses=n_classes, model_type=args.model_type)
            saved_model_name = "./models/trained_models/" + args.dataset + "_" + args.model_type + "_" + args.version +".pt"
            model.load_state_dict(torch.load(saved_model_name))
            model = create_AAA_model(model, args.dataset, args.batch_size, args.model_type)
            evaluate_model(model, args.dataset, model_type=args.model_type)            
        elif args.defense == 'Spline':
            model = get_model(nclasses=n_classes, model_type=args.model_type)
            saved_model_name = "./models/trained_models/" + args.dataset + "_" + args.model_type + "_" + args.version +".pt"
            model.load_state_dict(torch.load(saved_model_name))
            model = create_splines_model(model, args.dataset, args.batch_size)
            evaluate_model(model, args.dataset)
        elif args.defense == 'MDTS':
            model = get_model(nclasses=n_classes, model_type=args.model_type)
            saved_model_name = "./models/trained_models/" + args.dataset + "_" + args.model_type + "_" + args.version +".pt"
            model.load_state_dict(torch.load(saved_model_name))
            model = create_md_temp_scale_model(model, args.model_type, args.dataset, args.batch_size, n_classes, use_spectral=True)
            evaluate_model(model, args.dataset)
        elif args.defense=="CS":
            model = get_model(nclasses=n_classes, model_type=args.model_type)
            saved_model_name = "./models/trained_models/" + args.dataset + "_" + args.model_type + "_" + args.version +".pt"
            model.load_state_dict(torch.load(saved_model_name))
            model = create_compressed_model(model, args.dataset, args.batch_size)
        elif args.defense=="TS":
            model = get_model(nclasses=n_classes, model_type=args.model_type)
            saved_model_name = "./models/trained_models/" + args.dataset + "_" + args.model_type + "_" + args.version +".pt"
            model.load_state_dict(torch.load(saved_model_name))
            model = create_temp_scale_model(model, args.dataset, args.batch_size)




