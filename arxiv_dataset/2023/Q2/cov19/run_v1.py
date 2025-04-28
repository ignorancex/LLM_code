#
# This revised script treats validation set as unseen images
#
# 
# BS=128; IMGSET='all2'; MID='wideres101'; exec( open('icassp_sep.py').read() )
if 0:    
    IMGSET='all2';     
    for BS in [16 ]:  
        for MID in [ 'vgg' ]:                    
            exec( open('icassp_sep.py').read() )            
                            
import torch
SEED=101
torch.manual_seed(SEED)

import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os, sys
import copy
from skimage import io, transform
from torchvision import datasets, models, transforms, utils
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

import scipy.ndimage as ndimage
import skimage.measure
import numpy as np

from torch.utils.data import Dataset
#import SimpleITK as sitk
#import pydicom as pyd
import logging
from tqdm import tqdm

from myutils import *
from datetime import date
today = date.today()
today = today.strftime("%Y-%m-%d")

import pandas as pd
import subprocess  # for quering file counts  quickly
from glob import glob

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
from torch.utils.data import Dataset, DataLoader

if ('val_partition' in globals())==False:
    
    print("PyTorch Version: ",torch.__version__) # 1.13.1 
    print("Torchvision Version: ", torchvision.__version__ )# ptorch-gpu 0.14.1
    
    val_partition = pd.read_excel( '~/scratch/ICASSP_severity_validation_partition.xlsx') # engine='openpyxl' )
    trn_partition = pd.read_excel( '~/scratch/ICASSP_severity_train_partition.xlsx' ) # engine='openpyxl' )

num_classes = 4

if ( 'tk' in globals())==False:
    tk=1
if ( 'MID' in globals())==False: # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    MID='inception' 
else:
    MID = MID.lower()   
if ( 'BS' in globals())==False:
    BS = 16
if ( 'EP' in globals())==False:        
    EP = 500    
if ( 'PRELOAD' in globals())==False:        
    PRELOAD = 1    
    Results = {}
    Results2= {}
    Details ={}    
if ( 'LR' in globals())==False:        
    LR = 0.001    
if ( 'OP' in globals())==False:        
    OP = 'adam'    

# Flag for feature extracting. When False, we finetune the whole model,
# When True we only update the reshaped layer params
feature_extract = True
 
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False            
            
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    is_inception=0
    
    if model_name == "vtb32":
        model_ft = models.vit_b_32( weights = torchvision.models.ViT_B_32_Weights.DEFAULT )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.heads.head.in_features
        print( model_name, 'has', num_ftrs, 'units on the last layer' )
        model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "wideres101":
        model_ft = models.wide_resnet101_2( weights = torchvision.models.Wide_ResNet101_2_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        print( model_name, 'has', num_ftrs, 'units on the last layer' )
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152( weights= torchvision.models.ResNet152_Weights.DEFAULT ) 
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        print( model_name, 'has', num_ftrs, 'units on the last layer' )
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet( weights = torchvision.models.AlexNet_Weights.DEFAULT  )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        print( model_name, 'has', num_ftrs, 'units on the last layer' )
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn( weights= torchvision.models.VGG11_BN_Weights.DEFAULT )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        print( model_name, 'has', num_ftrs, 'units on the last layer' )
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(weights =torchvision.models.squeezenet.SqueezeNet1_0_Weights.DEFAULT )
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet201":
        """ Densenet
        """
        model_ft = models.densenet201(weights= torchvision.models.DenseNet201_Weights.DEFAULT )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        print( model_name, 'has', num_ftrs, 'units on the last layer' )
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(weights= torchvision.models.DenseNet121_Weights.DEFAULT )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        print( model_name, 'has', num_ftrs, 'units on the last layer' )
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3( weights= torchvision.models.Inception_V3_Weights.DEFAULT  )
        set_parameter_requires_grad(model_ft, feature_extract)
        
        is_inception=1
        
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        print( model_name, 'has', num_ftrs, 'units on the last layer of auxilary net' )
        
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        print( model_name, 'has', num_ftrs, 'units on the last layer of primary net' )
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    print('\n\n******************\n', MID , '\n\n\n\n')
    #model_ft = model_ft.float() 
    return model_ft, input_size, is_inception 

 
model, input_size, is_inception = initialize_model( MID, num_classes, feature_extract, use_pretrained=True)
print('\n\n',input_size)            

if 1:
    print("Initializing Datasets and Dataloaders...")
    
    from torch.utils.data import DataLoader
    class CTDataset_in_ram( Dataset ):
        """CT dataset"""
        def __init__(self, all_scans, TID, csv_file, dataframe=None, debug=False ):        
            
            if  isinstance( csv_file, list):
                self.list = csv_file 
                self.islist=1
            else:
                self.islist=0
                self.list = pd.read_csv(csv_file)         
            
            self.scan_names = []
            self.debug = debug 
            
            self.all_scans = all_scans                      
            self.dataframe =dataframe
            
            self.descriptions = []
                        
            self.scan_nums=[]                
                
            assert len(all_scans) == len(csv_file)                       
            
        def __len__(self):
            return len(self.list)
        
        def __getitem__(self, idx):        

            if torch.is_tensor(idx):
                idx = idx.tolist()    
       
            if self.islist:
                ff=self.list[idx]
            else:
                ff=self.list.iloc[idx].values[0]
                 
            a = ff.split('_')                   
            self.scan_nums.append(  int( a[-1] ) )      
            self.scan_names.append( ff )      
            
            labels =  np.zeros( num_classes )                            
            l = np.nan
            if self.dataframe is not None:                
                l = self.dataframe.loc[ self.dataframe ['Name'] == ff, 'Category'].values[0] -1  # starts from zero so deduct 1                                    
                labels[l] = 1                         
                
            sample = self.all_scans[ idx, ]
                        
            if self.debug:
                    print(sample.shape, labels, idx ) 
                    
            return (sample,labels)

    if PRELOAD:        
        all_scans = {}
        all_scan_nums = {}
        tids = ['train', 'val', 'test' ]                
            
        for tid in tids:  
            filename ='~scratch/numpy/%s_%d_%s_mid.npy.npz'%( IMGSET, input_size,tid )
            d=np.load( filename )
            print( '\n\n**********\n', filename )
                
            if tid=='train':
                all_scans[tid]=d['a'][:460, ]
            else:
                all_scans[tid]=d['a']
            all_scan_nums[tid]=d['b']        
        
        ds = {}
        
        duplicate_cate4 = trn_partition.loc[  trn_partition.Category == 4, :]        
        list_trn = list( duplicate_cate4.Name.values ) + list( trn_partition.Name.values ) 
        
        Ntrains = 430
        VAL=5
        val_inds = np.arange( 0, Ntrains, VAL )
        trn_inds = np.setdiff1d( np.arange(Ntrains) , val_inds  )

        list_trn = list( trn_partition.Name.values[trn_inds] ) 
        list_val = list( trn_partition.Name.values[val_inds] ) 
        list_val2= list( val_partition.Name.values ) 
        
        
ds['train'] = CTDataset_in_ram( all_scans = all_scans['train'][trn_inds, ], csv_file = list_trn, dataframe=trn_partition, debug=False, TID = 'train' )
ds['val']   = CTDataset_in_ram( all_scans = all_scans['train'][val_inds, ], csv_file = list_val, dataframe=trn_partition, debug=False, TID = 'val' )
ds['val2']  = CTDataset_in_ram( all_scans=all_scans['val'], csv_file=list_val2, dataframe=val_partition, debug=True, TID = 'val' )

filename ='~/scratch/severity_test.csv' # partial list used before email (list got created before all uploading was done)
filename ='~/scratch/test/test_mar19.csv' # entire list per email
list_tst= pd.Series(  pd.read_csv( filename, header = None )[0].values ).tolist()      
print( 'train:', len(trn_inds), 'train2',  len(val_inds), len(list_tst), 'test')
ds['test']  = CTDataset_in_ram( all_scans=all_scans['test'], csv_file=list_tst, dataframe=None, debug=True, TID = 'test' )

for tid in tids:
    print( 'check for empty slice', np.sum(np.sum(np.sum(all_scans[tid],2),1),1) ); 
    print( np.where( np.sum(np.sum(np.sum(all_scans[tid],2),1),1)  ==0) )
            
# Create training and validation dataloaders
dataloaders = {x: torch.utils.data.DataLoader( ds[x], batch_size=BS, shuffle=True) for x in ['train', 'val']}
dataloaders['val2'] = torch.utils.data.DataLoader( ds['val2'], batch_size=1, shuffle=False)
dataloaders['test'] = torch.utils.data.DataLoader( ds['test'], batch_size=1, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

params_to_update = model.parameters()

print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
if OP=='SGD':
    optimizer = optim.SGD(params_to_update, lr=LR, momentum=0.9)
else:
    optimizer = optim.Adam(params_to_update, lr=LR )    
     
PATH = '~s/cratch/mdls_sep/severity_%s_%s_BS%d_%s_LR%.3f' % (IMGSET, MID, BS, OP, LR)
print( '\n\n\nWriting results to', PATH , '\n\n' ) 

# Setup the loss f x n
criterion = nn.CrossEntropyLoss()

since = time.time()
val_acc_history = []
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

def early_stopping(train_loss, validation_loss, min_delta, tolerance):
    # https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    counter = 0
    if (validation_loss - train_loss) > min_delta:
        counter +=1
        if counter >= tolerance:
            return True
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

if tk==1:
    TK = [1,2]
else:
    TK = [tk]
        
for MODE in TK:    
    if MODE==1:
        early_stopper = EarlyStopper(patience=5, min_delta=8)
        val_loss = []
        trn_loss= []

        for epoch in range(0,EP):
            print('Epoch {}/{}'.format(epoch, EP - 1))
            print('-' * 10)

            for phase in ['train', 'val']:          

                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   

                running_loss = 0.0
                running_corrects = 0                       
                batch_count = -1

                for inputs, labels in dataloaders[phase]: 
                    batch_count+=1   

                    inputs = inputs.to(device) # BS * 3 * input_size * input_size                   
                    labels = labels.to(device) # BS * num_class                   

                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled( phase == 'train'):                       

                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)

                            # BS x 4 classes, BS x 4 classes,  labels                              
                            # print( outputs.shape, aux_outputs.shape, labels.shape ) 
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            #print( outputs.shape) 
                            loss = criterion(outputs, labels)

                        #print( phase, batch_count, 'loss:', loss)
                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum( preds == torch.max(labels,1)[1]  )  # torch.max returns max_value, indices

                N= len(dataloaders[phase].dataset) 
                epoch_loss = running_loss / (1e-10+N)
                epoch_acc = running_corrects.double() /(1e-10+N)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                try:
                    ls = loss.to('cpu').numpy().item()
                except:
                    # trn phase
                    ls = loss.to('cpu').detach().numpy().item()
                    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,            
                    }, PATH + '.pkl' )
                    
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                    val_loss.append( ls )
                else:
                    trn_loss.append( ls )

            if early_stopper.early_stop(val_loss[-1] ):                         
                #if early_stopping(epoch_train_loss, epoch_validate_loss)
                break
            elif epoch >early_stopper.patience:
                print( early_stopper.counter, early_stopper.patience, 'last 3 validation losses', val_loss[-3:]  )


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)    
        
    elif MODE==3:                             
        
        if torch.cuda.is_available():
            checkpoint = torch.load(PATH + '.pkl')            
        else:
            checkpoint = torch.load(PATH + '.pkl', map_location=torch.device('cpu') )
            
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']            
        print( 'model loaded')   
        
if tk:        
    model.eval()
    
    Pred={}    
    Actual={}
    
    dataloaders = {x: torch.utils.data.DataLoader( ds[x], batch_size=1, shuffle=False) for x in ['train', 'val', 'val2', 'test']}

    tids = ['train', 'val', 'val2','test' ]
    for phase in tids:   

        Pred[phase]=[]
        Actual[phase]=[]
        batch_count = -1
                
        ds[phase].debug=False
        
        if phase != 'test':
            if 0:
                print('\n\n\n******************\nscan_id, predicted_category, actual_category' )
        else:
            print('\n\nscan_id, predicted_category' )
            
        for inputs, labels in dataloaders[phase]: 
            #print( ds[phase].scan_names[ -1 ] )

            batch_count+=1
            # ds[phase].scan_nums = []        

            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) 
            preds = preds.cpu().numpy()

            tt = torch.max(labels,1)[1].numpy()
            for i,p in enumerate( preds ):

                j=i +(batch_count) 
                sid=ds[phase].scan_names[ j ]
                
                Actual[phase].append( tt[i] )
                Pred[phase].append( p )
                 
                if phase=='test':
                    print( '%s, %d'% (sid, p ))         
     
    from sklearn.metrics import * #1_score
    phase = 'train'        
    print('\n--------------------\n',PATH, '\n--------------------')
    f1t=f1_score( Actual[phase], Pred[phase], average = 'macro') 
    f1t_s=phase + 'F1-macro=%.3f\n'% f1t    
    print( f1t_s, '\n' )        
    conf_t=confusion_matrix(Actual[phase] , Pred[phase])    
    print( conf_t )
    
    phase = 'val'
    f1v=  f1_score( Actual[phase], Pred[phase] , average = 'macro')
    f1v_s=phase + 'F1-macro=%.3f\n'% f1v    
    print( '\n\n', f1v_s,'\n' )
    conf_v =confusion_matrix(Actual[phase] , Pred[phase])
    print( conf_v )
    
    phase = 'val2'
    f1v2=  f1_score( Actual[phase], Pred[phase] , average = 'macro')
    f1v2_s= 'Unseen validation:' + 'F1-macro=%.3f'% f1v2    
    print( '\n\n', f1v2_s,'\n' )
    conf_v2 =confusion_matrix(Actual[phase] , Pred[phase])
    print( conf_v2)
        
    Results[ PATH ]= [f1t, f1v, f1v2 ] 
    Results2[ PATH ]= [conf_t, conf_v, conf_v2 ]     
    
    
if ( 'trn_loss'  in globals() ):    
    plt.close('all'); 
    plt.figure(figsize=(12,8))
    plt.plot( trn_loss, label='TRN' )
    plt.plot( val_loss ,label='VAL')  
    plt.text( 0,1, np.array2string( conf_v)  , weight='bold', fontsize=12 )
    plt.text( len(trn_loss),1, np.array2string( conf_v2)  , weight='bold', fontsize=12 )
    tit_str=f1t_s +  f1v_s + f1v2_s + '\n(conf matrix on val and left-out sets)'
    plt.title(tit_str)
    plt.legend()
    plt.savefig( '%s_progress.png'%PATH )

if ( 'trn_loss'  in globals() ):
    # write once
    Details[ PATH ] = '%d epochs, %s, %s' %( epoch, torch.__version__,  torchvision.__version__)
    write2pkl( PATH+'_scores' , {'Details': Details,'Results':Results, 'Results2':Results2, 'epoch':epoch,'Actual':Actual, 'Pred':Pred } )    
    del trn_loss

# exec( open('icassp_sep.py').read() ) 
# exec( open('get_res.py').read()  ) 

      
