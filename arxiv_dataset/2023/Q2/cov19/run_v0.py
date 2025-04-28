
# Adopted from 
# https://github.com/munniomer/pytorch-tutorials/blob/master/beginner_source/finetuning_torchvision_models_tutorial.py
#    
#

SEED=101
import torch
torch.manual_seed(SEED)

import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
from skimage import io, transform
from torchvision import datasets, models, transforms, utils
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import subprocess  # for quering file counts  quickly
from glob import glob

tids = ['train', 'val', 'test' ]      

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
from torch.utils.data import Dataset, DataLoader

num_classes = 4

if ( 'Results' in globals())==False:
    Results={}
    Results2={}
    
if ( 'MID' in globals())==False:
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    MID='inception' 
        
if ( 'BS' in globals())==False:
    BS = 16
if ( 'EP' in globals())==False:        
    EP = 200
if ( 'PRELOAD' in globals())==False:        
    PRELOAD = 1
    # exec( open('extract_and_store_slices.py').read()  )

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
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

    if model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152( weights= torchvision.models.ResNet152_Weights.DEFAULT ) 
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet( weights = torchvision.models.AlexNet_Weights.DEFAULT  )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn( weights= torchvision.models.VGG11_BN_Weights.DEFAULT )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
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

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(weights= torchvision.models.DenseNet121_Weights.DEFAULT )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
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
        
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()    
    
    return model_ft, input_size, is_inception 
  
model, input_size, is_inception = initialize_model( MID, num_classes, feature_extract, use_pretrained=True)


PATH = '~/scratch/mdls/severity_all2_%s_BS%d' % (MID,BS)
print( '\n\n\nWriting results to', PATH , '\n\n' ) 

class CTDataset_in_ram( Dataset ):
    """CT dataset"""
    def __init__(self, all_scans, TID, csv_file, dataframe=None, debug=False ):        

        if  isinstance( csv_file, list):
            self.list = csv_file 
            self.islist=1
        else:
            self.islist=0
            self.list = pd.read_csv(csv_file)         


        self.debug = debug 

        self.all_scans = all_scans            
        self.TID = TID            
        self.dataframe =dataframe

        self.descriptions = []

        self.scan_nums=[]                
        #self.resz_flip = transforms.Compose([transforms.ToTensor(), transforms.Resize(input_size), transforms.RandomHorizontalFlip(p=0) ])                
        #self.resz      = transforms.Compose([transforms.ToTensor(), transforms.Resize(input_size), ])                

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
        #print( '????', a[-1] )
        self.scan_nums.append(  int( a[-1] ) )      

        labels =  np.zeros( num_classes )                            
        l = np.nan
        if self.dataframe is not None:
            l = self.dataframe.loc[ self.dataframe ['Name'] == ff, 'Category'].values[0] -1  # starts from zero so deduct 1                                    
            labels[l] = 1                         

        sample = self.all_scans[ self.TID ][ idx, ]

        if self.debug:
                print(sample.shape, labels, idx ) 

        return (sample,labels)
 
from torch.utils.data import DataLoader

if PRELOAD:        
    all_scans = {}
    all_scan_nums = {}

    for tid in tids:
        d=np.load('~/scratch/all2_%d_%s_mid.npy.npz'%(input_size,tid ))
        all_scans[tid]=d['a']
        all_scan_nums[tid]=d['b']        

    ds = {}

    duplicate_cate4 = trn_partition.loc[  trn_partition.Category == 4, :]

    list_trn = list( duplicate_cate4.Name.values ) + list( trn_partition.Name.values ) 
    list_trn = list( trn_partition.Name.values ) 
    list_val = list( val_partition.Name.values ) 

    #all_scans, TID, csv_file, dataframe=None, debug=False
    ds['train'] = CTDataset_in_ram( all_scans=all_scans, csv_file=list_trn, dataframe=trn_partition, debug=True, TID = 'train' )
    ds['val']   = CTDataset_in_ram( all_scans=all_scans, csv_file=list_val, dataframe=val_partition, debug=True, TID = 'val' )
    
    fff ='~scratch/severity_test.csv' # 101 files only (upload not done properly)    
    fff ='~scratch/test/test_mar19.csv' # entire list per email on 2023-03-19

    list_tst=pd.read_csv( fff ).values               
    ds['test'] = CTDataset_in_ram( all_scans=all_scans, csv_file=list_val, dataframe=None, debug=True, TID = 'test' )

    print('Sanity checks')
    for tid in tids:
        print(all_scans[tid].sum())

                       
# Create training and validation dataloaders
dataloaders = {x: torch.utils.data.DataLoader( ds[x], batch_size=BS, shuffle=True) for x in ['train', 'val', 'test']}

ds['train'].debug = ds['val'].debug = True 
for phase in tids:
    next(iter( dataloaders[phase] ))            
ds['train'].debug = ds['val'].debug = False

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
# optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

optimizer = optim.Adam(params_to_update, lr=0.001 )

# Setup the loss f x n
criterion = nn.CrossEntropyLoss()



since = time.time()
val_acc_history = []
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

# option #1
def early_stopping(train_loss, validation_loss, min_delta, tolerance):
    # https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    counter = 0
    if (validation_loss - train_loss) > min_delta:
        counter +=1
        if counter >= tolerance:
            return True
       
# option #2    
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
        early_stopper = EarlyStopper(patience=5, min_delta=10)
        val_loss = []
        trn_loss= []

        for epoch in range(EP):
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
                    
                    #if batch_count >3:
                    #    break 
                    # inputs, labels = next(iter( dataloaders[phase] ))            

                    inputs = inputs.to(device) # BS x 3 x input_size x input_size                   
                    labels = labels.to(device) # BS x num_class                   

                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled( phase == 'train'):                       

                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)

                            # BS x4 classes, BS x 4 classes...                              
                            # print( outputs.shape, aux_outputs.shape, labels.shape ) 
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            #print( outputs.shape) 
                            loss = criterion(outputs, labels)

                        print( phase, batch_count, 'loss:', loss)
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

            if early_stopper.early_stop(val_loss[-1] ):  # early_stopping(epoch_train_loss, epoch_validate_loss)
                break
            elif epoch >early_stopper.patience:
                print( early_stopper.counter, early_stopper.patience, 'last 3 validation losses', val_loss[-3:]  )


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)                
        
        plt.close('all'); 
        plt.plot( trn_loss, label='trn' )
        plt.plot( val_loss ,label='val' )        
        plt.legend()
        plt.savefig( '%s_progress.png'%PATH )       
        
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
         
    for phase in tids:   

        Pred[phase]=[]
        Actual[phase]=[]
        batch_count = -1
                
        ds[phase].debug=False
        
        if phase != 'test':
            print('\n\n\n******************\nscan_id, predicted_category, actual_category' )
        else:
            print('\n\nscan_id, predicted_category' )
            
        for inputs, labels in dataloaders[phase]: 

            batch_count+=1            
            # ds[phase].scan_nums = []        

            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) 
            preds = preds.cpu().numpy()

            tt = torch.max(labels,1)[1].numpy()
            
            for i,p in enumerate( preds ):          
                j=i + BS*(batch_count) 
                sid=ds[phase].scan_nums[ j ]

                Actual[phase].append( tt[i] )
                Pred[phase].append( p )
                if phase == 'test':
                    print('test_ct_scan_%d, %d'% (sid, p ))
                else:
                    print('ct_scan_%d, %d, %d'% (sid, p, tt[i]  ))
         
     
    from sklearn.metrics import * #1_score
    phase = 'train'        
    print('\n--------------------')
    f1t=f1_score( Actual[phase], Pred[phase], average = 'macro') 
    print( phase, f1t )        
    conf_t=confusion_matrix(Actual[phase] , Pred[phase])    
    print( conf_t )
    
    phase = 'val'
    f1v=  f1_score( Actual[phase], Pred[phase] , average = 'macro')
    print( phase, f1v )
    conf_v =confusion_matrix(Actual[phase] , Pred[phase])
    print( conf_v )
    
    Results[ PATH ]= [f1t, f1v] 
    Results2[ PATH ]= [conf_t, conf_v] 
    
      

    

'''
--------------------
~/scratch/mdls2/severity_all2_resnet152_BS128_adam_LR0.008 
--------------------
trainF1-macro=1.000 

[[132   0   0   0]
 [  0 123   0   0]
 [  0   0 166   0]
 [  0   0   0  39]]


 valF1-macro=0.912 

[[29  0  2  0]
 [ 0 20  0  0]
 [ 1  0 44  0]
 [ 0  0  2  3]]
'''
