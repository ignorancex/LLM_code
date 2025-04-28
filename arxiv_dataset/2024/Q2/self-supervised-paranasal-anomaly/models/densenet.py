import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from functools import partial

from medcam import medcam

import pytorch_lightning as pl

from utils.utils_eval import _test_step, get_eval_dictionary, _test_end,get_eval_dictionary_classification, _test_step_multiclass
import hydra

import torch.optim as optim
import torchio as tio
from utils.utils_eval import  redFlagEvaluation_einscanner, calc_thresh, calc_thresh_classification,evaluateModel


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=2):

        super().__init__()

        # First convolution
        self.features = [('conv1',
                          nn.Conv3d(n_input_channels,
                                    num_init_features,
                                    kernel_size=(conv1_t_size, 7, 7),
                                    stride=(conv1_t_stride, 2, 2),
                                    padding=(conv1_t_size // 2, 3, 3),
                                    bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))
        self.features = nn.Sequential(OrderedDict(self.features))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out,
                                    output_size=(1, 1,
                                                 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def generate_model(model_depth, **kwargs):
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         **kwargs)
    elif model_depth == 169:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 32, 32),
                         **kwargs)
    elif model_depth == 201:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 48, 32),
                         **kwargs)
    elif model_depth == 264:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         **kwargs)

    return model



class DenseNetClassifier(pl.LightningModule):

    def __init__(self,cfg,prefix=None):
        super(DenseNetClassifier, self).__init__()
        

        self.cfg = cfg
        # Model 
        self.model = generate_model(model_depth=self.cfg.model_depth,num_classes=len(list(self.cfg.class_weights)))
        #print layer name for gradcam
        
        if self.cfg.enable_gradcam: 
            
            self.model = medcam.inject(self.model, output_dir=self.cfg.attention_folder, backend=self.cfg.method, layer='auto',label=1, save_maps=True)

        self.prefix =prefix
        self.softmax = torch.nn.Softmax(dim=1)
        # Loss function
        cls_weights = torch.tensor(list(self.cfg.class_weights)).float()
        self.criterion = torch.nn.CrossEntropyLoss(weight=cls_weights).cuda()
        
    

    def configure_optimizers(self):
        
        optimizer =  optim.Adam(self.model.parameters() ,lr=self.cfg.lr, amsgrad=False)
        scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,verbose=True )
        return {"optimizer": optimizer, "lr_scheduler": scheduler,"monitor": "val/loss"} #[optimizer],[scheduler]

    def prepare_batch(self, batch):
        
        return {"image": batch['one_image'][tio.DATA],
                "label": batch['label'],
                "disease_label": batch['disease_label'],
                "patient_disease_id":batch['patient_disease_id'],
                "image_path":batch['image_path'],
                "smax":batch['smax'],
                "crop_size":batch['crop_size'],} 
    

    
    
    def training_step(self, batch, batch_idx):

        return_object = self.prepare_batch(batch)
        
        inputs = return_object['image']

        # print(inputs.shape)
        target = return_object['label']

        
        
        outputs  = self.model(inputs)
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        #print(y, y.shape, outputs.shape)
        
        loss = self.criterion(outputs,target)
        self.log(f'train/loss',loss.item(), prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        return {"loss": loss}

    def on_validation_epoch_start(self): 

        if len(list(self.cfg.class_weights)) == 2:
            self.val_eval_dict = get_eval_dictionary_classification()
        else:
            self.val_eval_dict = []
            for i in range(len(list(self.cfg.class_weights))):
                self.val_eval_dict.append(get_eval_dictionary_classification().copy())

            
    
    def validation_step(self, batch, batch_idx):
        #self.counter += 1
        
        return_object = self.prepare_batch(batch)
        inputs = return_object['image']
        target = return_object['label'] 
        patient_disease_id = return_object['patient_disease_id']
        

        outputs = self.model(inputs)

        
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        
        loss = self.criterion(outputs,target)
        logits = self.softmax(outputs)

        #check if binary or multiclass
        
        if len(list(self.cfg.class_weights)) == 2:

            AnomalyScoreReco_vol = logits[:,1]
            self.val_eval_dict['labelPerVol'] = torch.cat((torch.tensor(self.val_eval_dict['labelPerVol']), target.cpu()), 0)
            self.val_eval_dict['AnomalyScorePerVol'] = torch.cat((torch.tensor(self.val_eval_dict['AnomalyScorePerVol']), AnomalyScoreReco_vol.cpu()), 0)

            # TODO Need to change the line below
            self.val_eval_dict['AnomalyScorePerVol_one_instance'] = torch.cat((torch.tensor(self.val_eval_dict['AnomalyScorePerVol_one_instance']), AnomalyScoreReco_vol.cpu()), 0)
            self.val_eval_dict['patient_disease_id'] = self.val_eval_dict['patient_disease_id'] + patient_disease_id
            
        else:   
            # Multi class
            for i in range(len(list(self.cfg.class_weights))): 

                AnomalyScoreReco_vol = logits[:,i] 

                target_class = torch.zeros_like(target)
                target_class[target == i] = 1 

                
                

                self.val_eval_dict[i]['labelPerVol'] = torch.cat((torch.tensor(self.val_eval_dict[i]['labelPerVol']), target_class.cpu()), 0) 
                self.val_eval_dict[i]['AnomalyScorePerVol'] = torch.cat((torch.tensor(self.val_eval_dict[i]['AnomalyScorePerVol']), AnomalyScoreReco_vol.cpu()), 0) 

                # TODO Need to change the line below
                self.val_eval_dict[i]['AnomalyScorePerVol_one_instance'] = torch.cat((torch.tensor(self.val_eval_dict[i]['AnomalyScorePerVol_one_instance']), AnomalyScoreReco_vol.cpu()), 0)
                self.val_eval_dict[i]['patient_disease_id'] = self.val_eval_dict[i]['patient_disease_id'] + patient_disease_id


        self.log('val/loss',loss.item(), prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        



        return {"loss": loss}

    def on_validation_epoch_end(self):

        #Calculate threshold 
        if len(list(self.cfg.class_weights)) == 2:
            thresh = calc_thresh_classification(self.val_eval_dict) 
            eval_dict = evaluateModel(self.val_eval_dict.copy(), thresh=thresh)

            self.log(f'val/F1',eval_dict['F1_thresh_1p_prc'], prog_bar=False, on_step=False, on_epoch=True)
        else:

            eval_dict = {}
            for i in range(len(list(self.cfg.class_weights))):

                thresh = calc_thresh_classification(self.val_eval_dict[i])
                eval_dict[i] = evaluateModel(self.val_eval_dict[i].copy(), thresh=thresh)
                eval_dict[i]['thresh'] = thresh

            # print F1 score for each class
            for i in range(len(list(self.cfg.class_weights))):
                self.log(f'val/F1_{i}',eval_dict[i]['F1_thresh_1p_prc'], prog_bar=False, on_step=False, on_epoch=True)
            #print(eval_dict)
        
        
        
        
        



    def on_test_start(self):
        if len(list(self.cfg.class_weights)) == 2:
            self.eval_dict = get_eval_dictionary_classification()
        else:
            # create a list of eval dictionaries which are not referenced to each other
            self.eval_dict = []
            for i in range(len(list(self.cfg.class_weights))):
                self.eval_dict.append(get_eval_dictionary_classification().copy())


        

    def test_step(self, batch, batch_idx: int):
        
        
        return_object = self.prepare_batch(batch)
        inputs = return_object['image']
        label = return_object['label']
        patient_disease_id = return_object['patient_disease_id']
        smax = return_object['smax']
        crop_size = return_object['crop_size']

        disease_target = return_object['disease_label']
       
        
        

        outputs = self.model(inputs)
        
        loss = self.criterion(outputs,label)
        target = self.softmax(outputs)
        
        if len(list(self.cfg.class_weights)) == 2:

            if target.shape[0] > 1: 
                
                AnomalyScoreReco_vol_mean = torch.mean(target[:,1]).item()
                AnomalyScoreReco_vol_std = torch.std(target[:,1]).item()

                

                AnomalyScoreReco_vol     = AnomalyScoreReco_vol_mean
                AnomalyScoreReco_vol_std = AnomalyScoreReco_vol_std 

            else: 

                AnomalyScoreReco_vol     = target[:,1].item()
                AnomalyScoreReco_vol_std = 0.0


            #AnomalyScoreReco_volL2 = 0
            
            
            self.eval_dict['AnomalyScorePerVol'].append(AnomalyScoreReco_vol)
            self.eval_dict['AnomalyScorePerVol_std'].append(AnomalyScoreReco_vol_std)
            
            
            #print(smax)
            self.eval_dict['smax'].append(smax[0])
            
            #Get anomaly score of the first instance 
            AnomalyScoreReco_vol_one_instance  = target[0,1].item()


            self.eval_dict['AnomalyScorePerVol_one_instance'].append(AnomalyScoreReco_vol_one_instance)

            _test_step(self, None,None,inputs,label[0],disease_target[0],patient_disease_id[0],crop_size[0],batch_idx) # everything that is independent of the model choice


        else:

            # Multi class
            for i in range(len(list(self.cfg.class_weights))):

                label_class = torch.zeros_like(label)
                label_class[label == i] = 1 

                #print(label_class, "label_class")  
                #print(label,"label")

                if target.shape[0] > 1: 
                    
                    AnomalyScoreReco_vol_mean = torch.mean(target[:,i]).item()
                    AnomalyScoreReco_vol_std = torch.std(target[:,i]).item()

                    

                    AnomalyScoreReco_vol     = AnomalyScoreReco_vol_mean
                    AnomalyScoreReco_vol_std = AnomalyScoreReco_vol_std 

                else: 

                    AnomalyScoreReco_vol     = target[:,i].item()
                    AnomalyScoreReco_vol_std = 0.0


                #AnomalyScoreReco_volL2 = 0
                
                
                self.eval_dict[i]['AnomalyScorePerVol'].append(AnomalyScoreReco_vol)
                self.eval_dict[i]['AnomalyScorePerVol_std'].append(AnomalyScoreReco_vol_std)
                
                
                #print(smax)
                self.eval_dict[i]['smax'].append(smax[0])
                
                #Get anomaly score of the first instance 
                AnomalyScoreReco_vol_one_instance  = target[0,i].item()
                self.eval_dict[i]['AnomalyScorePerVol_one_instance'].append(AnomalyScoreReco_vol_one_instance)
                
                # everything that is independent of the model choice
                _test_step_multiclass(self, None,None,inputs,label_class[0],disease_target[0],patient_disease_id[0],crop_size[0],batch_idx,index=i) 


        
            #print("ENDDDDDD")
            #print("Class 0",self.eval_dict[0]['labelPerVol'],self.eval_dict[0]['diseaseLabelPerVol'])
            #print("Class 1",self.eval_dict[1]['labelPerVol'],self.eval_dict[1]['diseaseLabelPerVol'])
            #print("Class 2",self.eval_dict[2]['labelPerVol'],self.eval_dict[2]['diseaseLabelPerVol']) 

        
            
        
        # calculate metrics
        
        