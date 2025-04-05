# -*- coding:utf-8 -*-
'''
Aum Sri Sai Ram
Implementation of CCT: Consensual Collaborative Training for Facial Expression Recognition with Noisy Annotations for ABAW2021        
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 01-07-2021
Email: darshangera@sssihl.edu.in
'''

import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse, sys
import datetime

from PIL import Image

from algorithm.noisyfer_extended_affwild2 import noisyfer
import pandas as pd
import image_utils
import cv2
import argparse,random

import pickle
import random
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')

parser.add_argument('--affwild2_path', type=str, default='../data/Affwild2/', help='Affwild2 dataset path.')

parser.add_argument('--testmetafile', type=str, default = '../data/Affwild2/Annotations_21/test_set.pkl', help='path to test list')


parser.add_argument('--metafile', type=str, default = '../data/Affwild2/Annotations_21/annotations_21.pkl', help='path to training list')
                       
parser.add_argument('--pretrained', type=str, default='pretrained/res18_naive.pth_MSceleb.tar', help='Pretrained weights')


parser.add_argument('--resume', type=str, default='', help='Use FEC trained models')
                         
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.4)

parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)

parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')

                  
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='affwild2')

parser.add_argument('--noise_file', type=str, help='EmoLabel/', default='../data/Affwild2/Annotations_21/annotations_21.pkl')#0.4noise_train.txt

 
parser.add_argument('--co_lambda_max', type=float, default=.9,  help='..based on ')
                    
parser.add_argument('--beta', type=float, default=.65,  help='..based on ')                    

parser.add_argument('--n_epoch', type=int, default=30)

parser.add_argument('--num_classes', type=int, default=7)

parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--print_freq', type=int, default=30)

parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')

parser.add_argument('--batch_size', type=int, default=128, help='batch_size')

parser.add_argument('--num_iter_per_epoch', type=int, default=400)

parser.add_argument('--epoch_decay_start', type=int, default=80)

parser.add_argument('--gpu', type=int, default=0)


parser.add_argument('--adjust_lr', type=int, default=1)

parser.add_argument('--num_models', type=int, default=3)

parser.add_argument('--model_type', type=str, help='[mlp,cnn,res]', default='res')

parser.add_argument('--save_model', type=str, help='save model?', default="False")

parser.add_argument('--save_result', type=str, help='save result?', default="True")

parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')

parser.add_argument('--warm_epochs', type=int, default=0, help='warm_epochs')

parser.add_argument('--test_results_generation', type=int, default=0, help='do test set evalaution')

parser.add_argument('--margin', type=float, default=0.7, help='margin for relbaelling')

parser.add_argument('--normalized', type=int, default=0, help='use normalized weights and features')

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
if args.gpu is not None:
    device = torch.device('cuda:{}'.format(args.gpu))
    torch.cuda.manual_seed(args.seed)

else:
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

# Hyper Parameters
batch_size = args.batch_size
learning_rate = args.lr


def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def switch_expression(expression_argument):
    switcher = {
         0:'Neutral',
         1:'Happiness',
          2: 'Sadness',
        3: 'Surprise',
4: 'Fear', 5: 'Disgust', 6: 'Anger',
    }
    return switcher.get(expression_argument, 0) #default neutral expression

def change_emotion_label_same_as_affectnet(emo_to_return):
        """
        Parse labels to make them compatible with AffectNet.  
        #https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/model/utils/udata.py
        """

        if emo_to_return == 0:
            emo_to_return = 0
        elif emo_to_return == 1:
            emo_to_return = 6
        elif emo_to_return == 2:
            emo_to_return = 5
        elif emo_to_return == 3:
            emo_to_return = 4
        elif emo_to_return == 4:
            emo_to_return = 1
        elif emo_to_return == 5:
            emo_to_return = 2
        elif emo_to_return == 6:
            emo_to_return = 3

        return emo_to_return



def _read_path_label(_train_mode, file_path):
        
        data = pickle.load(open(file_path, 'rb'))
        # read frames ids
        if _train_mode == 'Train':
            data = data['EXPR_Set']['Train_Set'] #changed from Training_Set to this
        elif _train_mode == 'Validation':
            data = data['EXPR_Set']['Validation_Set']
        elif _train_mode == 'Test':
            data = data['EXPR_Set']['Test_Set']
        else:
            raise ValueError("train mode must be in : Train, Validation, Test")
        return data

def default_reader(fileList,  num_classes=7, train_mode ='Train'):
    imgList = []
   
    start_index = 1
    if train_mode =='Train':
       max_samples = 500000000
    else:
       max_samples = 900000000
    num_per_cls_dict = dict()
    for i in range(0, num_classes):
        num_per_cls_dict[i] = 0
    
    expression_0, expression_1,expression_2, expression_3, expression_4,expression_5,expression_6 = 0,0,0,0,0,0,0
    data_dict = _read_path_label(train_mode, fileList)

    #print(train_mode, len(data_dict.keys()))
    
    all_list = []         
    if  train_mode in ['Train','Validation']: #training or validation
       if train_mode =='Validation':
            for video_name in data_dict.keys(): #Each video is a key            
                frame_dict  = data_dict[video_name]
                #print(frame_dict.keys())#keys: ['label', 'path', 'frames_ids']
                labels, imagepaths = frame_dict['label'],frame_dict['path']
                for i in range(len(labels)):
                    imagename,label = imagepaths[i][2:],labels[i] #2: is to remove ./ in ./cropped_aligned/48-30-720x1280/01589.jpg 0
                    all_list.append([imagename,label])
                    #print(len(all_list), all_list[-3:])
       elif train_mode =='Train':
            for video_name in data_dict.keys(): #Each video is a key            
                frame_dict  = data_dict[video_name]
                #print(frame_dict.keys())#keys: ['label', 'path', 'frames_ids']
                labels, imagepaths = frame_dict['label'],frame_dict['path']
                for i in range(len(labels)):
                    imagename,label = imagepaths[i][2:],labels[i] #2: is to remove ./ in ./cropped_aligned/48-30-720x1280/01589.jpg 0
                    all_list.append([imagename,label])
                    #print(len(all_list), all_list[-3:]) 
            #Adding below valid to train
            """
            data_dict = _read_path_label('Validation', fileList)
            for video_name in data_dict.keys(): #Each video is a key            
                frame_dict  = data_dict[video_name]
                #print(frame_dict.keys())#keys: ['label', 'path', 'frames_ids']
                labels, imagepaths = frame_dict['label'],frame_dict['path']
                for i in range(len(labels)):
                    imagename,label = imagepaths[i][2:],labels[i] #2: is to remove ./ in ./cropped_aligned/48-30-720x1280/01589.jpg 0
                    all_list.append([imagename,label])
                    #print(len(all_list), all_list[-3:])
            """
    elif train_mode =='Test': #test
         for video_name in data_dict.keys(): #Each video is a key            
                #if video_name =='218':
                frame_dict  = data_dict[video_name]
                #print(frame_dict.keys())#keys: ['label', 'path', 'frames_ids']
                labels, imagepaths = frame_dict['label'], frame_dict['path']
                for i in range(len(labels)):
                    imagename,label = imagepaths[i][2:],labels[i] #2: is to remove ./ in ./cropped_aligned/48-30-720x1280/01589.jpg 0
                    all_list.append([imagename,label])
    else:
           print('Not implemented yet.\n')
    #print(len(all_list),all_list[:2], all_list[-3:]) 
    random.shuffle(all_list)
    #print(len(all_list),all_list[:2], all_list[-3:]) 

    for i in range(len(all_list)):
            imgPath, expression =all_list[i]

            expression = change_emotion_label_same_as_affectnet(expression) 

            if expression == 0:
               expression_0 = expression_0 + 1            
               if expression_0 > max_samples:
                  continue
  
            if expression == 1:
               expression_1 = expression_1 + 1
               if expression_1 > max_samples:
                  continue  

            if expression == 2:
               expression_2 = expression_2 + 1
               if expression_2 > max_samples:
                  continue  

            if expression == 3:
               expression_3 = expression_3 + 1
               if expression_3 > max_samples:
                  continue  

            if expression == 4:
               expression_4 = expression_4 + 1
               if expression_4 > max_samples:
                  continue  

            if expression == 5:
               expression_5 = expression_5 + 1
               if expression_5 > max_samples:
                  continue  

            if expression == 6:
               expression_6 = expression_6 + 1
               if expression_6 > max_samples:
                  continue  

            imgList.append([imgPath, expression])
            num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1 
        
    print(train_mode, ' has total included: ', len(imgList), ' with split \t', num_per_cls_dict)
    return imgList,num_per_cls_dict

   
class ImageList(data.Dataset):
    def __init__(self, root, fileList, train_mode = 'Validation', transform=None,  list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.cls_num = 7
        self.imgList, self.num_per_cls_dict =  list_reader(fileList, self.cls_num, train_mode)
        self.transform = transform
        self.loader = loader
        self.fileList  = fileList
        self.train_mode = train_mode
        self.label = [ self.imgList[i][1] for i in range(len(self.imgList)) ]
    def __getitem__(self, index):
        imgPath,  target_expression = self.imgList[index]
        
        imagefullpath = os.path.join(self.root, imgPath)

        if not os.path.exists(imagefullpath) and self.train_mode == 'Test':
           return None, imgPath  , index       

        face = self.loader(os.path.join(self.root, imgPath))       
        
        label = target_expression 
        if self.transform is not None:
            face = self.transform(face)
       
        if self.train_mode == 'Test':
           return  face, imgPath , index
        else:
           return  face, target_expression , index
    def __len__(self):
        return len(self.imgList)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


                                      
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        #print(self.indices)    
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        #print(self.num_samples)              
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            #print(label)
            # spdb.set_trace()
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        #print(dataset_type)
        #pdb.set_trace()
        if dataset_type is ImageList:
            return dataset.imgList[idx][1]
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples  
     
def create_test_output(filename='ExprChallenge_21/test_predictions.csv'):
    d = dict()
    with open(filename,'r') as fp:
         lines = fp.readlines()#.sort()
         lines.sort()
         lines = [line.strip().split('/')[1:3] for line in lines]
         print('\ntotal: ', len(lines), lines[:2])#,lines[-20:])
          
         for line in lines:
             key,value = line[0], line[1]
             video_file = 'ExprChallenge_21/'+key+'.txt'
             if not os.path.exists(video_file):
                
                f = open(video_file,'w')
                f.write('Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise')
             else:
                
                f = open(video_file,'a') 
             
             #print(key, len(value), type(value), value[:5])

             
             name, emotion = value.replace("'","").split(',')#v.replace("'","").split(',')[0], v.replace("'","").split(',')[1]
             #print(key,value, name, emotion)
             label  = emotion#switch_expression(emotion) #only here because emotion name is written
             
             f.write('\n'+str(label))
             #print('\n'+name+' '+str(label))
             f.close()
             

    print('\nTest out created.')

def main():
    
    print('\n\t\tAum Sri Sai Ram')
    print('\n\tFER on Affwild2 2021 ABAW using CCT \n')
    print(args)
    #print('\nloading dataset...')
    
    print('\n\nMeta file', args.noise_file)   
    #args.noise_file = noise_file
    
    if  args.dataset == 'affwild2':   
        input_channel = 3
        num_classes = 7
        init_epoch = 5
        args.epoch_decay_start = 100
        
        filter_outlier = False
        args.model_type = "res"
        data_transforms = transforms.Compose([            
            transforms.RandomHorizontalFlip(p=0.5),        
            transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
                            transforms.RandomAffine(degrees=0, translate=(.1, .1),
                                                   scale=(1.0, 1.25),
                                                   resample=Image.BILINEAR)],p=0.5),
            
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            
            ])
            
                                    
        
        data_transforms_val = transforms.Compose([         
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
                                     
        train_dataset = ImageList(root=args.affwild2_path, fileList = args.metafile, train_mode='Train',
                  transform=data_transforms)
        
    
        print('\n Train set size:', train_dataset.__len__()) 
                                                                                   
        test_dataset = ImageList(root = args.affwild2_path, fileList = args.metafile, train_mode='Validation', transform=data_transforms_val)   
        
        print('\n Validation set size:', test_dataset.__len__())
        
        
        if args.forget_rate is None:
            forget_rate = args.noise_rate
        else:
            forget_rate = args.forget_rate   
            
            
        train_sampler = None#ImbalancedDatasetSampler(train_dataset)  
             
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)    
                                                   
                                  
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = batch_size,
                                               num_workers = args.num_workers,
                                               shuffle = False,  
                                               pin_memory = True)                                      
    # Define models
    print('building model...'+ str(args.num_models)+' with noise file: '+args.noise_file)
    
    
    model = noisyfer(args, train_dataset, device, input_channel, num_classes)
    epoch = 0
    train_acc1 = 0
    train_acc2 = 0
    
    best_acc = 0.0
    best_expr_f1 = 0
    final_cm = 0
    final_mcm = 0
    best_prec1 = 0
    acc_list = []
    
    #Only testset evaluation
    if args.test_results_generation:
       test_dataset = ImageList(root = args.affwild2_path, fileList = args.testmetafile, train_mode='Test', transform=data_transforms_val)
       test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = batch_size,
                                               num_workers = args.num_workers,
                                               shuffle = False,  
                                               pin_memory = True) 

       dest_file = 'ExprChallenge_21/test_predictions.csv'                                               
       model.test_results_generation(test_loader, dest_file)
       create_test_output(dest_file)
       print('Sairam. Done')
       assert False
    #exiting 
    
    
    # training
    for epoch in range(0, args.n_epoch+1):
        if args.num_models == 1:
           acc = model.train(train_loader, epoch)
           acc = model.evaluate(test_loader)
           print(  'Epoch [%d/%d] Test Accuracy on the %s test images: %% Avg Accuracy %.4f' % ( epoch + 1, args.n_epoch, len(test_dataset), acc))
           
        elif args.num_models == 2:
             train_acc1, train_acc2, acc = model.train(train_loader, epoch)
             acc1, acc2, acc, prec1, f1, cm = model.evaluate(test_loader)
             print("Epoch: {}   Validation Acc: {}, Validation f1:{} and Final score :{}".format(epoch, prec1, f1, 0.0033*prec1+0.67*f1))
             
        elif args.num_models == 3:
             
             train_acc1, train_acc2, train_acc3, acc = model.train(train_loader, epoch)
             test_acc1, test_acc2, test_acc3, acc, prec1, f1, cm = model.evaluate(test_loader)
             print("Epoch: {}   Validation Acc: {}, Validation f1:{} and Final score :{}".format(epoch, prec1, f1, 0.0033*prec1+0.67*f1))
             print(  'Epoch [%d/%d] Validation Acc on the %s test images: Model1 %.4f %% Model2 %.4f %% Model3 %.4f %% Avg Accuracy %.4f' % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2,  test_acc3,acc))
             
            
             model.save_model(epoch, str(0.0033*prec1+0.67*f1), prec1, f1  )
        
        elif args.num_models == 4:
             train_acc1, train_acc2, train_acc3, train_acc4, acc = model.train(train_loader, epoch)
             test_acc1, test_acc2, test_acc3, test_acc4, acc,prec1, f1, cm  = model.evaluate(test_loader)
             print("Epoch: {}   Validation Acc: {}, Validation f1:{} and Final score :{}".format(epoch, prec1, f1, 0.0033*prec1+0.67*f1))
             print( 'Epoch [%d/%d] Validation Accuracy on the %s  Validation images: Model1 %.4f %% Model2 %.4f %% Model3 %.4f %%  Model4 %.4fAvg Accuracy %.4f' % ( epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2,  test_acc3, test_acc4,acc))
                    
                    
             model.save_model(epoch, str(0.0033*prec1+0.67*f1), prec1, f1  )
        
 

        

if __name__ == '__main__':
    main()
    