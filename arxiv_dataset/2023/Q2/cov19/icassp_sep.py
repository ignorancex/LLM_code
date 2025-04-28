#    Copyright 2023 lisatwyw Lisa Y.W. Tang 
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


# cd /home/your_userid/scratch/; BS=128;  
# 
# IMGSET='all2'; NORM=2; MID='squeezenet'; tk=3; FT=1; exec( open('icassp23/icassp_sep.py').read() )
#

if 0:    
    IMGSET='all2';     
    for BS in [32]:  
        feature_extract = False  
        for MID in sorted( [ 'wideres101', 'inception', 'squeezenet', 'resnet152', 'densenet201', 'densenet', 'alexnet', 'vgg', 'vtb32' ]):                    
            exec( open('icassp_sep.py').read() )            
                        
            
import torch
SEED=101
torch.manual_seed(SEED)

import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms, utils
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

print( 'Torch', torch.__version__, 'Torchvision', torchvision.__version__ )
import numpy as np
import matplotlib.pyplot as plt
import time
import os, sys, copy

from skimage import io, transform
import skimage.measure
import scipy.ndimage as ndimage


import logging
from tqdm import tqdm

from icassp23.myutils import *

from datetime import date
today = date.today()
today = today.strftime("%Y-%m-%d")

import pandas as pd
import subprocess  # for quering file counts quickly
from glob import glob
   

if ('val_partition' in globals())==False:
    
    print("PyTorch Version: ",torch.__version__) # 1.13.1 
    print("Torchvision Version: ", torchvision.__version__ )# ptorch-gpu 0.14.1

    val_partition = pd.read_excel( 'icassp23/ICASSP_severity_validation_partition.xlsx') # engine='openpyxl' )
    trn_partition = pd.read_excel( 'icassp23/ICASSP_severity_train_partition.xlsx' ) # engine='openpyxl' )
    
num_classes = 4            
MIDnames=[ 'wideres101', 'inception', 'squeezenet', 'resnet152', 'densenet201', 'densenet', 'alexnet', 'vgg', 'vtb32' ]     
    
for i in sys.argv:
    print(i)
try:
    ctn=1    
    MID = int( sys.argv[ctn] ); ctn+=1
    FT = int( sys.argv[ctn] ); ctn+=1
    BS = int( sys.argv[ctn] ); ctn+=1
    OP = sys.argv[ctn]; ctn+=1
    NORM = int(sys.argv[ctn]); ctn+=1
    
    MID = MIDnames[ MID ]
except:
    pass
    
if NORM:
    OUTDIR = 'mdls_sep_norm'
else:
    OUTDIR = 'mdls_sep'; 

try:
    os.mkdir( '%s/'% OUTDIR )
except:
    pass


if ( 'tk' in globals())==False:
    tk=1
if ( 'MID' in globals())==False: # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    MID='inception' 
else:
    MID = MID.lower()
    
if ( 'BS' in globals())==False:
    BS = 16
if ( 'EP' in globals())==False:        
    EP = 250

if ( 'IMGSET' in globals())==False:        
    IMGSET= 'all2'
if ( 'PRELOAD' in globals())==False:        
    PRELOAD = 1    
    
    
Results = {}
Results1 = {}
Results2 = {}
Results3 = {}
Results4 = {}
Results5 = {}
Results6 = {}
Details ={}
    
if ( 'LR' in globals())==False:        
    LR = 0.001    
if ( 'OP' in globals())==False:        
    OP = 'adam'    
if ( 'FT' in globals())==False:            
    FT =1

if FT:
    feature_extract=False
else:
    feature_extract=True
           
            
model, input_size, is_inception = myutils.initialize_model( MID, num_classes, feature_extract, use_pretrained=FT>0 )
print('\n\n',input_size)            
 
print("Initializing Datasets and Dataloaders...")


xform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize([0.45], [0.224]) ])      

import Augmentor
p = Augmentor.Pipeline()
p.random_distortion(.4, 4, 4, 8)


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
        self.resz_flip = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5) ])                
        #self.resz      = transforms.Compose([transforms.ToTensor(), transforms.Resize(input_size), ])                

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
        sample[0,0,0] =sample[1,0,0] =sample[2,0,0]= 0 # reset this label

        if NORM==1:
            tform = xform                
        elif NORM==2:
            tform = self.resz_flip
        elif NORM==3:
            tform = self.resz_flip
        if NORM:    
            s1=xform(sample[0,])
            s2=xform(sample[1,])
            s3=xform(sample[2,])

            sample[0,] = s1  
            sample[1,] = s2
            sample[2,] = s3
        #np.stack( ( s1[0,], s2[0,], s3[0,]), axis=0 )    
        
        if self.debug:
                #print('reset')
                print(sample.shape, labels, idx ) 

        return (sample,labels)

if PRELOAD:        
    all_scans = {}
    all_scan_nums = {}
    tids = ['train', 'val', 'test' ]                

    for tid in tids:                        

        filename ='/home/your_userid/scratch/icassp23/numpy/%s_%d_%s_mid.npy.npz'%( IMGSET, input_size,tid )
        d=np.load( filename )
        print( '\n\n**********\n', filename )

        if tid=='train':
            all_scans[tid]=d['a'][:460, ]
        else:
            all_scans[tid]=d['a']
        all_scan_nums[tid]=d['b']        

    ds = {}
    duplicate_cate4 = trn_partition.loc[  trn_partition.Category == 4, :]

    #list_trn = list( duplicate_cate4.Name.values ) + list( trn_partition.Name.values ) 

    Ntrains = 430
    VAL=4
    val_inds = np.arange( 0, Ntrains, VAL )
    trn_inds = np.setdiff1d( np.arange(Ntrains) , val_inds  )

    list_trn = list( trn_partition.Name.values[trn_inds] ) 
    list_trn_val = list( trn_partition.Name.values[val_inds] ) 
    list_val2= list( val_partition.Name.values )        

for tid in tids:
    #print( 'check for empty slice', np.where( np.sum(np.sum(np.sum(all_scans[tid][],2),1),1)), all_scans[tid].shape )          
    print( 'check for empty slice',)#  np.sum(np.sum(np.sum(all_scans[tid],2),1),1) ); 
    print( np.where( np.sum(np.sum(np.sum(all_scans[tid],2),1),1)  ==0) )
    
for t in tids: # [ 'val2']: 
    print( '\t\t',t, 'intensity range before processing by CTdataset_in_RAM: \t\t', all_scans[t][0,1,:,:].min() , all_scans[t][0,1,:,:].max() ) 
print( '(4 in the training set is possible because label encoded; e.g. IMGSET=\'all2\')' )




ds['train'] = CTDataset_in_ram( all_scans = all_scans['train'][trn_inds, ], csv_file = list_trn,     dataframe=trn_partition, debug=False, TID = 'train' )
ds['val']   = CTDataset_in_ram( all_scans = all_scans['train'][val_inds, ], csv_file = list_trn_val, dataframe=trn_partition, debug=False, TID = 'val' )
ds['val2']  = CTDataset_in_ram( all_scans = all_scans['val'],               csv_file = list_val2,    dataframe=val_partition, debug=False, TID = 'val2' )

fff ='/home/your_userid/scratch/icassp23/test/test_mar19.csv' # entire list per email

list_tst= pd.Series( pd.read_csv( fff, header = None )[0].values ).tolist()      
print( '|train|=', len(trn_inds), '|train2|=',  len(val_inds), '|test|=', len(list_tst))
ds['test']   = CTDataset_in_ram( all_scans=all_scans['test'], csv_file=list_tst, dataframe=None, debug=False, TID = 'test' )
          
# Create training and validation dataloaders
dataloaders = {x: torch.utils.data.DataLoader( ds[x], batch_size=BS, shuffle=False) for x in ['train', 'val']}
dataloaders['val2'] = torch.utils.data.DataLoader( ds['val2'], batch_size=1, shuffle=False)
dataloaders['test'] = torch.utils.data.DataLoader( ds['test'], batch_size=1, shuffle=False)

print('\n\nImage statistics:')            

labels = {}
for tid in [ 'train']:# 'val','val2' ]:
    labels[ tid ] = [] #np.empty(0)
    print( 'Batching through unseen validation set:')
    for i,a in enumerate( dataloaders[tid]):
        if i==0:
            aaa = a[0]
        labels[tid].append( np.argmax( a[1].numpy()) )
    print( tid, ': n = %02d' % len( labels[tid] ), '\t\t# of samples per category:', np.histogram(labels[tid], [0,1,2,3,4])[0] )
 
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
     
if NORM:
    PATH = '%s/%s_%s_FT%d_BS%d_%s_LR%.3f_NRM%d' % (OUTDIR, IMGSET, MID, FT, BS, OP, LR, NORM )
else:
    PATH = '%s/%s_%s_FT%d_BS%d_%s_LR%.3f' % (OUTDIR, IMGSET, MID, FT, BS, OP, LR)
print( '\n\nPATH = ', PATH , '\n\n' ) 

# Setup the loss f x n
criterion = nn.CrossEntropyLoss()

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
        since = time.time()
        val_acc_history = []
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        early_stopper = EarlyStopper(patience=5, min_delta=10)
        val_loss = []
        trn_loss= []

        for epoch in range(0,EP):
            # print('Epoch {}/{}'.format(epoch, EP - 1)); print('-' * 10)

            for tid in ['train', 'val']:          

                if tid == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   

                running_loss = 0.0
                running_corrects = 0                       
                batch_count = -1

                for inputs, labels in dataloaders[tid]: 
                    batch_count+=1
                    
                    #if batch_count >3:
                    #    break 
                    # inputs, labels = next(iter( dataloaders[tid] ))            

                    inputs = inputs.to(device) # BS * 3 * input_size * input_size                   
                    labels = labels.to(device) # BS * num_class                   

                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled( tid == 'train'):                       

                        if is_inception and tid == 'train':
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

                        #print( tid, batch_count, 'loss:', loss)
                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training tid
                        if tid == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum( preds == torch.max(labels,1)[1]  )  # torch.max returns max_value, indices

                N= len(dataloaders[tid].dataset) 
                epoch_loss = running_loss / (1e-10+N)
                epoch_acc = running_corrects.double() /(1e-10+N)
          
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(tid, epoch_loss, epoch_acc), end='.' )

                try:
                    ls = loss.to('cpu').numpy().item()
                except:
                    # trn tid
                    ls = loss.to('cpu').detach().numpy().item()
                    
                # deep copy the model
                if tid == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,            
                    }, PATH + '.pkl' )
                    
                if tid == 'val':
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
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
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
    for tid in tids:   
        Pred[tid]=[]
        Actual[tid]=[]
        batch_count = -1
                
        ds[tid].debug=False
        
        if tid != 'test':
            if 0:
                print('\n\n\n******************\nscan_id, predicted_category, actual_category' )
        else:
            print('\n\nscan_id, predicted_category' )
            
        for inputs, labels in dataloaders[tid]: 
            #print( ds[tid].scan_names[ -1 ] )

            batch_count+=1
            # ds[tid].scan_nums = []        

            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) 
            preds = preds.cpu().numpy()

            tt = torch.max(labels,1)[1].numpy()
            for i,p in enumerate( preds ):

                j=i +(batch_count) 
                sid=ds[tid].scan_names[ j ]
                
                Actual[tid].append( tt[i] )
                Pred[tid].append( p )
                 
                if tid=='test':
                    print( '%s, %d'% (sid, p ))         
     
    from sklearn.metrics import * #1_score
    from sklearn.preprocessing import OneHotEncoder
    
    print('\n--------------------\n',PATH, '\n--------------------')
    f1scores = {}
    f1scores_s = {}
    confs,mc,rc,pr= {},{}, {},{} 
    enc= OneHotEncoder(); 
    
    Results2[ PATH ] = []
    Results3[ PATH ] = []
    Results4[ PATH ] = []
    Results5[ PATH ] = []
    Results6[ PATH ] = []
    
    f1scores_pairwise = {}  

    for tid in ['train', 'val', 'val2' ]:
        print()
        f1scores[tid] = f1 = f1_score( Actual[tid], Pred[tid], average = 'macro') 
                        
        f1scores_pairwise[tid] = np.zeros(4)
        for c in range(4):        
            f1scores_pairwise[tid][c] =f1_score(np.asarray(Actual[tid] )+1== c, np.asarray(Pred[tid])+1==c, average = 'macro') 
                
        f1scores_s[tid] = tid + ': F1-macro=%.1f | '% (f1 *100 )      
        print( f1scores_s[tid], end=' | ' )                
        
        print('F1-macro-pairwise:', np.array2string( 100*f1scores_pairwise[tid], precision=1, formatter={'float_kind':lambda x: "%.1f" % x}   ))
        
        confs[tid] = confusion_matrix(Actual[tid] , Pred[tid])    
        print( confs[tid], '\n' )    
   
        pr[tid]= precision_score(  Actual[tid], Pred[tid], average='macro' )
        rc[tid]= recall_score(  Actual[tid], Pred[tid], average='macro' )
        mc[tid]= matthews_corrcoef(Actual[tid], Pred[tid] )
                   
        p1=np.asarray(Actual[tid]).reshape(-1,1)
        p1=enc.fit_transform( p1 ).toarray() 
        p2=np.asarray(Pred[tid]).reshape(-1,1)
        p2=enc.fit_transform( p2 ).toarray() 
        
        Results6[ PATH ].append(  roc_auc_score( p1, p2, multi_class='ovr' ) )
                             
    Results[ PATH ]= [f1scores['train'], f1scores['val'], f1scores['val2'] ] 
    Results2[ PATH ]= [confs['train'], confs['val'], confs['val2'] ]     
    
    Results4[ PATH ]= [pr['train'], pr['val'], pr['val2'] ] 
    Results3[ PATH ]= [rc['train'], rc['val'], rc['val2'] ] 
    
    Results5[ PATH ]= [mc['train'], mc['val'], mc['val2'] ]         
    
    print(np.histogram(Pred['test'] ,[0,1,2,3,4])[0] )
        
if ( 'trn_loss'  in globals() ):    
    
    plt.close('all'); 
    plt.figure(figsize=(12,8))
    plt.plot( trn_loss, label='TRN' )
    plt.plot( val_loss ,label='VAL')  
    
    plt.text( 0,1, np.array2string( confs['val'])  , weight='bold', fontsize=12 )
    plt.text( len(trn_loss),1, np.array2string( confs['val2'] )  , weight='bold', fontsize=12 )
    tit_str= f1scores_s['train']  +  f1scores_s['val'] + f1scores_s['val2'] + '\n(conf matrix on val and left-out sets)'
    plt.title(tit_str)
    plt.ylabel( 'Loss' )
    plt.xlabel( np.histogram(Pred['test'] ,[0,1,2,3,4])[0] )
    plt.legend()
    plt.savefig( '%s_progress.png'%PATH )

 
    # write once
    Details[ PATH ] = '%d epochs, %s, %s' %( epoch, torch.__version__,  torchvision.__version__)
    write2pkl( PATH+'_scores' , {'Details': Details, 'Results':Results, 'Results2':Results2,\
                                 'Results3':Results3, 'Results4':Results4,\
                                 'Results5':Results5, 'Results6':Results6, 'epoch':epoch, 'Actual':Actual, 'Pred':Pred } )    
    del trn_loss

    # write2pkl( '/home/your_userid/scratch/actuallabels', {'Actual':Actual})
 
