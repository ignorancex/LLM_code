import argparse
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import sklearn.metrics as metrics

from torch.utils.data import Dataset, DataLoader
import albumentations as A
import os
import random
import yaml

import utils
from roi import *
import preprocessing as pre
import architectures

def parse_args():
    parser = argparse.ArgumentParser(description='roi demo') 

    parser.add_argument('--model_name', default='model_primary', type=str, help='Name of the model/experiment') 
    parser.add_argument('--model_ssl_rgb', default=None, type=str, help='Pretrained model name for rgb branch') 
    parser.add_argument('--model_ssl_flow', default=None, type=str, help='Pretrained model name for rgb branch') 
    parser.add_argument('--model_ssl_rgb_flow', default=None, type=str, help='Pretrained model name for rgb and flow branch') 
    parser.add_argument('--clip_frames', default=16, type=int, help='Number of frames to consider for each prediction') 
    parser.add_argument('--eval', action=argparse.BooleanOptionalAction)

    parsed_args = parser.parse_args()

    assert not ((parsed_args.model_ssl_rgb or parsed_args.model_ssl_flow) and parsed_args.model_ssl_rgb_flow) # Cannot load pretrained weights of the same branches twice
        
    return parsed_args

parsed_args = parse_args()


is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU.
if is_cuda:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    
torch.manual_seed(8)
random.seed(8)
np.random.seed(8)



class ConfigObject:
    def __init__(self, dictionary):
        self.__dict__ = dictionary

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join("config/", config_name)) as file:
        config = yaml.safe_load(file)

    return config

args = load_config("primary.yaml")
args = ConfigObject(args)

# Adding parsed_args parameters
args.model_name              = parsed_args.model_name
args.model_self_supervised   = {"rgb": parsed_args.model_ssl_rgb, 
                            "flow": parsed_args.model_ssl_flow, 
                            "rgb_and_flow": parsed_args.model_ssl_rgb_flow} 
args.clip_frames             = parsed_args.clip_frames
args.eval                    = parsed_args.eval
args.device                  = device

"""
class Set_Parameters():
    model_name          = parsed_args.model_name #"model_XYZ"
    model_self_supervised = {"rgb": parsed_args.model_ssl_rgb, 
                             "flow": parsed_args.model_ssl_flow, 
                             "rgb_and_flow": parsed_args.model_ssl_rgb_flow} #None # "model_RGB_FLOW_VICReg_IJK"
    eval                = parsed_args.eval
    coclr               = False
    mixed_precision     = True
    clip_frames         = parsed_args.clip_frames # Number of frames to consider for each prediction
    interval            = 4
    batch_size          = 32
    dropout             = 0.2 
    dropout3d           = 0.2 #0.2
    epochs              = 30
    patience            = 15 # Number of epochs without improvement to tolerate
    
    lr                  = 0.01
    t_max               = epochs # T_max: maximum number of iterations. Parameter of the CosineAnnealing
    eta_min             = 0.001
    diff_lr_rgb         = False
    diff_lr_flow        = False
    
    
    device              = device
    
    dataset             = "RWF-2000" 
    fps                 = 7.5

args = Set_Parameters()
"""

class Paths():
    jpg_frames             = "datasets/RWF-2000_frames"
    models                 = "models/primary"
    models_self_supervised = "models/auxiliary"
paths = Paths()


n_of_frames = 16
   
if(args.dataset == "RWF-2000"):
    segments, labels = pre.create_window_segments_and_labels (paths, args)
    x_train = segments['train']
    train_labels = labels['train']
    x_val = segments['val']
    val_labels = labels['val'] 


class Augmentation():    
    rgb = A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                ], additional_targets={'image': 'image', 
                                       'image0': 'image', 
                                       'image1': 'image', 
                                       'image2': 'image', 
                                       'image3': 'image', 
                                       'image4': 'image', 
                                       'image5': 'image', 
                                       'image6': 'image', 
                                       'image7': 'image', 
                                       'image8': 'image',
                                       'image9': 'image', 
                                       'image10': 'image', 
                                       'image11': 'image', 
                                       'image12': 'image', 
                                       'image13': 'image', 
                                       'image14': 'image'}
    )
    rgb_and_flow = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(224,224)
                ], additional_targets={'image': 'image', 
                                       'image0': 'image', 
                                       'image1': 'image', 
                                       'image2': 'image', 
                                       'image3': 'image', 
                                       'image4': 'image', 
                                       'image5': 'image', 
                                       'image6': 'image', 
                                       'image7': 'image', 
                                       'image8': 'image',
                                       'image9': 'image', 
                                       'image10': 'image', 
                                       'image11': 'image', 
                                       'image12': 'image', 
                                       'image13': 'image', 
                                       'image14': 'image', 
                                       'flow': 'image', 
                                       'flow0': 'image', 
                                       'flow1': 'image', 
                                       'flow2': 'image', 
                                       'flow3': 'image', 
                                       'flow4': 'image', 
                                       'flow5': 'image', 
                                       'flow6': 'image', 
                                       'flow7': 'image', 
                                       'flow8': 'image',
                                       'flow9': 'image', 
                                       'flow10': 'image', 
                                       'flow11': 'image', 
                                       'flow12': 'image', 
                                       'flow13': 'image', 
                                       'flow14': 'image'}
    )
augm = Augmentation()


class Standardization():    
    rgb = A.Compose([
                A.Normalize (mean=(105.5504, 103.1739, 101.9839), 
                             std=(60.9979, 60.6278, 61.1028), 
                             always_apply=True,
                             max_pixel_value=255.0) 
                            ])
    flow = A.Compose([
                A.Normalize (mean=(-2.3771e-08, -2.8841e-08), 
                             std=(22.2633, 18.8949), 
                             always_apply=True,
                             max_pixel_value=1.0) 
                            ])
standardization = Standardization()

# https://en.wikipedia.org/wiki/Standard_score
# We can normalize both rgb and flow by assuming that each channel represent the same quantity,
# thus the normalization can be applied for all the channels together
def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    if (std == 0):
        return data-mean
    return (data-mean) / std


class Preprocess():    
    resize = A.Compose([
                    A.Resize(224,224)
                    ])
preprocess = Preprocess()



class Create_Dataset(Dataset):
    def __init__(self, video_names, labels, args, standardization, augm=None):

        self.labels = torch.tensor(np.array(labels)).float()
        self.video_names = video_names
        self.num_samples = len(self.video_names)
        self.args = args
        self.augm = augm
        self.standardization = standardization

        
    def __len__(self):
        return self.num_samples    
        
    def __getitem__(self, idx):
        
        flow_segment = []
        rgb_segment = []
        for i,name in enumerate(self.video_names[idx]):
            with open(self.video_names[idx][i],'rb') as f: 
                frame = cv2.imdecode(np.frombuffer(f.read(),dtype=np.uint8), -1)
                rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
                rgb_segment.append(rgb)

            if (i == 0):
                prev = frame
            else:
                nexT = frame
                flow = pre.generate_flow (prev, nexT)
                flow_segment.append(flow)
                prev = nexT

        flow_segment.append(np.zeros((224,224,2)))

        # Data augmentation
        ## First apply augmentation only on RGB
        if (self.augm):
            augmented = self.augm.rgb(image=rgb_segment[0], 
                                  image0=rgb_segment[1], 
                                  image1=rgb_segment[2],
                                  image2=rgb_segment[3],
                                  image3=rgb_segment[4],
                                  image4=rgb_segment[5],
                                  image5=rgb_segment[6],
                                  image6=rgb_segment[7],
                                  image7=rgb_segment[8],
                                  image8=rgb_segment[9],
                                  image9=rgb_segment[10],
                                  image10=rgb_segment[11],
                                  image11=rgb_segment[12],
                                  image12=rgb_segment[13],
                                  image13=rgb_segment[14],
                                  image14=rgb_segment[15])
            rgb_segment = [value for _, value in augmented.items()]  
            
            ## Then apply augmentation on both RGB and flow
            augmented = self.augm.rgb_and_flow(image=rgb_segment[0], 
                                  image0=rgb_segment[1], 
                                  image1=rgb_segment[2],
                                  image2=rgb_segment[3],
                                  image3=rgb_segment[4],
                                  image4=rgb_segment[5],
                                  image5=rgb_segment[6],
                                  image6=rgb_segment[7],
                                  image7=rgb_segment[8],
                                  image8=rgb_segment[9],
                                  image9=rgb_segment[10],
                                  image10=rgb_segment[11],
                                  image11=rgb_segment[12],
                                  image12=rgb_segment[13],
                                  image13=rgb_segment[14],
                                  image14=rgb_segment[15],
                                  flow=flow_segment[0],
                                  flow0=flow_segment[1],
                                  flow1=flow_segment[2],
                                  flow2=flow_segment[3],
                                  flow3=flow_segment[4],
                                  flow4=flow_segment[5],
                                  flow5=flow_segment[6],
                                  flow6=flow_segment[7],
                                  flow7=flow_segment[8],
                                  flow8=flow_segment[9],
                                  flow9=flow_segment[10],
                                  flow10=flow_segment[11],
                                  flow11=flow_segment[12],
                                  flow12=flow_segment[13],
                                  flow13=flow_segment[14],
                                  flow14=flow_segment[15])
            augmented = list(augmented.items()) # Transform the dictionary into list to capture its order
            rgb_segment = [value for _, value in augmented[:16]]
            flow_segment = [value for _, value in augmented[16:]]
            #rgb_segment = channel_replication(rgb_segment)
        
        # flow_segment = [standardization.flow(image=value)['image'] for value in flow_segment]
        # Padding based on region of interest
        #for i in range(args.clip_frames):
            #rgb_segment[i] = roi_pad (rgb_segment[i], flow_segment[i])
            #rgb_segment[i] = roi_pad_square (rgb_segment[i], flow_segment[i])
            #rgb_segment[i] = roi (rgb_segment[i], flow_segment[i])
        rgb_segment = roi_video(rgb_segment, flow_segment)
        #rgb_segment, flow_segment_new = roi_video_and_flow(rgb_segment, flow_segment)
        
        # Standardization
        #rgb_segment = [standardization.rgb(image=value)['image'] for value in rgb_segment]
        rgb_segment  = normalize(rgb_segment)            
        #flow_segment = [standardization.flow(image=value)['image'] for value in flow_segment]
        flow_segment = normalize(flow_segment)

        # From list to Numpy and Tensor
        rgb_segment = torch.FloatTensor(np.transpose(np.array(rgb_segment), (3,0,1,2)))
        flow_segment = torch.FloatTensor(np.transpose(np.array(flow_segment), (3,0,1,2)))  
        
        item = {'rgb': rgb_segment,
                'flow': flow_segment,
                'label': self.labels[idx]}   
        return item
    
def generate_subset (data, percentage):
    np.random.seed(8)
    idxs = np.random.randint(0, len(data), int(len(data)*percentage))
    return idxs

train_loader = DataLoader(Create_Dataset(x_train, train_labels, args, standardization, augm=augm),
                                batch_size=args.batch_size, num_workers = 1, shuffle=True, drop_last=True)
val_loader = DataLoader(Create_Dataset(x_val, val_labels, args, standardization, augm=None),
                            batch_size=args.batch_size, num_workers = 1, shuffle=False, drop_last=False)
model_rgb = architectures.FGN_RGB(args)
model_flow = architectures.FGN_FLOW(args)
model_merge_classify = architectures.FGN_MERGE_CLASSIFY(args)


if (args.coclr):
    
    #### INITIALIZE MODEL FOR ADDITIONAL SSL METHODS ####
    if (args.model_self_supervised["rgb"] ):
        checkpoint = torch.load(os.path.join(paths.models_self_supervised, args.model_self_supervised["rgb"]+".tar"), map_location=torch.device('cpu'))
        print (model_rgb)


        new_dict = {}
        for k,v in checkpoint['state_dict'].items():
            if ('encoder_q.0' in k):
                k = k.replace('encoder_q.0.', '')
                new_dict[k] = v

        state_dict = new_dict
        model_rgb.load_state_dict(state_dict, strict=True)

    if (args.model_self_supervised['flow']):      
        checkpoint = torch.load(os.path.join(paths.models_self_supervised, args.model_self_supervised['flow']+".tar"), map_location=torch.device('cpu'))
        print (model_flow)

        new_dict = {}
        for k,v in checkpoint['state_dict'].items():
            if ('encoder_q.0' in k):
                k = k.replace('encoder_q.0.', '')
                new_dict[k] = v

        state_dict = new_dict
        model_flow.load_state_dict(state_dict, strict=True)
        
    model = architectures.FGN(model_rgb, model_flow, model_merge_classify)
    model = model.to(args.device)
    model = nn.DataParallel(model).to(args.device)

else:
    if (args.model_self_supervised["rgb"]):
        model_rgb = utils.load_sub_network(model_rgb, args, paths, input_type="rgb", model_name="model")
    if (args.model_self_supervised["flow"]):
        model_flow = utils.load_sub_network(model_flow, args, paths, input_type="flow", model_name="model")
    if (args.model_self_supervised["rgb_and_flow"]):
        model_rgb = utils.load_sub_network(model_rgb, args, paths, input_type="rgb_and_flow", model_name="model_rgb")
        model_flow = utils.load_sub_network(model_flow, args, paths, input_type="rgb_and_flow", model_name="model_flow")
        #model_merge_classify = utils.load_sub_network(model_merge_classify, args, paths, input_type="rgb_and_flow", model_name="merging_block")

    model = architectures.FGN(model_rgb, model_flow, model_merge_classify)
    model = model.to(args.device)
    model = nn.DataParallel(model)

if (args.diff_lr_rgb):
    lr_rgb =  args.lr/10
    opt_rgb = optim.SGD([
        {"params": model.module.model_rgb.conv1.parameters(), "lr": lr_rgb/(2.6**3)},
        {"params": model.module.model_rgb.conv2.parameters(), "lr": lr_rgb/(2.6**3)},
        {"params": model.module.model_rgb.conv3.parameters(), "lr": lr_rgb/(2.6**2)},
        {"params": model.module.model_rgb.conv4.parameters(), "lr": lr_rgb/(2.6**2)},
        {"params": model.module.model_rgb.conv5.parameters(), "lr": lr_rgb/2.6},
        {"params": model.module.model_rgb.conv6.parameters(), "lr": lr_rgb/2.6},        
        {"params": model.module.model_rgb.conv7.parameters(), "lr": lr_rgb},
        {"params": model.module.model_rgb.conv8.parameters(), "lr": lr_rgb},
        {"params": model.module.model_rgb.batchnorm1.parameters(), "lr": lr_rgb},
        {"params": model.module.model_rgb.batchnorm2.parameters(), "lr": lr_rgb},
        {"params": model.module.model_rgb.batchnorm3.parameters(), "lr": lr_rgb},
        {"params": model.module.model_rgb.batchnorm4.parameters(), "lr": lr_rgb},
        {"params": model.module.model_rgb.batchnorm5.parameters(), "lr": lr_rgb},
        {"params": model.module.model_rgb.batchnorm6.parameters(), "lr": lr_rgb},
        {"params": model.module.model_rgb.batchnorm7.parameters(), "lr": lr_rgb},
        {"params": model.module.model_rgb.batchnorm8.parameters(), "lr": lr_rgb}
        ],
        lr=lr_rgb, momentum=0.9, weight_decay=1e-4, nesterov=True)
    eta_min_rgb = 0
else:
    opt_rgb = optim.SGD(model.module.model_rgb.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    eta_min_rgb = args.eta_min
    
if (args.diff_lr_flow):   
    lr_flow =  args.lr/10
    opt_flow = optim.SGD([
        {"params": model.module.model_flow.conv1f.parameters(), "lr": lr_flow/(2.6**3)},
        {"params": model.module.model_flow.conv2.parameters(), "lr": lr_flow/(2.6**3)},
        {"params": model.module.model_flow.conv3.parameters(), "lr": lr_flow/(2.6**2)},
        {"params": model.module.model_flow.conv4.parameters(), "lr": lr_flow/(2.6**2)},
        {"params": model.module.model_flow.conv5.parameters(), "lr": lr_flow/2.6},
        {"params": model.module.model_flow.conv6.parameters(), "lr": lr_flow/2.6},        
        {"params": model.module.model_flow.conv7.parameters(), "lr": lr_flow},
        {"params": model.module.model_flow.conv8.parameters(), "lr": lr_flow},
        {"params": model.module.model_flow.batchnorm1.parameters(), "lr": lr_flow},
        {"params": model.module.model_flow.batchnorm2.parameters(), "lr": lr_flow},
        {"params": model.module.model_flow.batchnorm3.parameters(), "lr": lr_flow},
        {"params": model.module.model_flow.batchnorm4.parameters(), "lr": lr_flow},
        {"params": model.module.model_flow.batchnorm5.parameters(), "lr": lr_flow},
        {"params": model.module.model_flow.batchnorm6.parameters(), "lr": lr_flow},
        {"params": model.module.model_flow.batchnorm7.parameters(), "lr": lr_flow},
        {"params": model.module.model_flow.batchnorm8.parameters(), "lr": lr_flow}
        ],
        lr=lr_flow, momentum=0.9, weight_decay=1e-4, nesterov=True)
    eta_min_flow = 0
else:
    opt_flow = optim.SGD(model.module.model_flow.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    eta_min_flow = args.eta_min
    
#opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
opt_merge = optim.SGD(model.module.model_merge.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

# Define the schedulers
scheduler_rgb = CosineAnnealingLR(opt_rgb, T_max=args.t_max, eta_min=eta_min_rgb)
scheduler_flow = CosineAnnealingLR(opt_flow, T_max=args.t_max, eta_min=eta_min_flow)
scheduler_merge = CosineAnnealingLR(opt_merge, T_max=args.t_max, eta_min=args.eta_min)
#scheduler = CosineAnnealingLR(opt, T_max=args.t_max, eta_min=args.eta_min)


CELoss = torch.nn.CrossEntropyLoss()
import warnings
warnings.filterwarnings("ignore")


if (not args.eval):
    train_loss_tot = []
    valid_loss_tot = []

    best_val_loss, best_val_epoch = None, 0
    best_metric = 0
    best_epoch = 0
        

    for epoch in tqdm(range(best_epoch, args.epochs)):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
            
        train_true = torch.empty(0).to(args.device)
        train_pred = torch.empty(0).to(args.device)
        train_logits = torch.empty(0).to(args.device)
        
        if (args.mixed_precision): scaler = torch.cuda.amp.GradScaler() 
        
        for step, s in enumerate(train_loader):  
            data_rgb = s['rgb'].to(args.device)
            data_flow = s['flow'].to(args.device)
            label = s['label'].to(args.device)           
            
            batch_size = data_rgb.size()[0]
            opt_rgb.zero_grad()
            opt_flow.zero_grad()
            opt_merge.zero_grad()

            
            if (args.mixed_precision):
                with torch.cuda.amp.autocast():
                    logits = torch.squeeze(model(data_rgb, data_flow)) 
                    
                    loss = F.binary_cross_entropy_with_logits(logits, label)
                scaler.scale(loss).backward()
                scaler.step(opt_rgb)
                scaler.step(opt_flow)
                scaler.step(opt_merge)
                scaler.update()
            else:
                logits = torch.squeeze(model(data_rgb, data_flow)) 
               
                loss = F.binary_cross_entropy_with_logits(logits, label)
                loss.backward()
                opt.step()
                
            preds = (logits>0).float()
            count += batch_size
            train_loss += loss.item() * batch_size
            
            train_true = torch.cat((train_true, label))                        
            train_pred = torch.cat((train_pred, preds))
            train_logits = torch.cat((train_logits, logits))

        
        
        train_loss = train_loss*1.0/count
        train_loss_tot.append(train_loss)  
        train_true = train_true.cpu().numpy().astype(int)
        train_pred = train_pred.detach().cpu().numpy().astype(int)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        train_f1 = metrics.f1_score(train_true, train_pred, average='macro')
        train_precision = metrics.precision_score(train_true, train_pred)
        train_recall = metrics.recall_score(train_true, train_pred)
        outstr = '\nTrain %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f \n train f1 score: %.6f, train precision: %.6f, train recall: %.6f' % (epoch,
                                                                                train_loss,
                                                                                train_acc,
                                                                                avg_per_class_acc,
                                                                                train_f1,
                                                                                train_precision,
                                                                                train_recall)

        print(outstr)
        
        
        
        
        
        
        ####################
        # Validation
        ####################
        val_loss = 0.0
        count = 0.0
        model.eval()
        val_true = torch.empty(0).to(args.device)
        val_pred = torch.empty(0).to(args.device)
        val_logits = torch.empty(0).to(args.device)
    
        
        with torch.no_grad():
            if (args.mixed_precision): scaler = torch.cuda.amp.GradScaler() 
                
            for step, s in enumerate(val_loader):
                data_rgb = s['rgb'].to(args.device)
                data_flow = s['flow'].to(args.device)
                label = s['label'].to(args.device)           
            
                batch_size = data_rgb.size()[0]
                
                if (args.mixed_precision):
                    with torch.cuda.amp.autocast():
                        logits = torch.squeeze(model(data_rgb, data_flow)) 
                        
                        loss = F.binary_cross_entropy_with_logits(logits, label)

                else:
                    logits = torch.squeeze(model(data_rgb, data_flow)) 
                   
                    loss = F.binary_cross_entropy_with_logits(logits, label)


                preds = (logits>0).float() 
                count += batch_size
                val_loss += loss.item() * batch_size
                val_true = torch.cat((val_true, label))                        
                val_pred = torch.cat((val_pred, preds))
                val_logits = torch.cat((val_logits, logits))


            scheduler_rgb.step()
            scheduler_flow.step()
            scheduler_merge.step()
            val_loss = val_loss*1.0/count
            valid_loss_tot.append(val_loss)
            val_true = val_true.cpu().numpy().astype(int)
            val_pred = val_pred.detach().cpu().numpy().astype(int)
            val_acc = metrics.accuracy_score(val_true, val_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(val_true, val_pred)
            val_f1 = metrics.f1_score(val_true, val_pred, average='macro')
            val_precision = metrics.precision_score(val_true, val_pred, average='macro')
            val_recall = metrics.recall_score(val_true, val_pred, average='macro')
            outstr = '\nVal %d, loss: %.6f, val acc: %.6f, val avg acc: %.6f \n val f1 score: %.6f, val precision: %.6f, val recall: %.6f' % (epoch,
                                                                                    val_loss,
                                                                                    val_acc,
                                                                                    avg_per_class_acc,
                                                                                    val_f1,
                                                                                    val_precision,
                                                                                    val_recall)
            
            print(outstr)
            
            
            # Early stopping
            metric = val_f1 #val_acc for action recognition
            if metric >= best_metric:
                best_metric = metric
                if not os.path.exists(paths.models): os.makedirs(paths.models)
                torch.save(model.state_dict(), os.path.join(paths.models, args.model_name+".pt"))
                print('----- Model saved -----')
            if (best_val_loss is None or val_loss < best_val_loss):
                best_val_loss, best_val_epoch = val_loss, epoch
                print ("----- Validation Loss decreased! -----")
            if best_val_epoch < epoch + 1 - args.patience:
                # nothing is improving for a while
                print ("----- Early stopping -----")
                break  

    #save_to_drive_folder(paths, "models", os.path.join(paths.models, args.model_name+'.pt'), args.model_name+'.pt')
    utils.plot_loss_function(args, paths, train_loss_tot, valid_loss_tot, epoch, save=False)
    print('--- End ---')
else:
    print ("----- Evaluation -----")



test_acc = 0.0
count = 0.0
val_true = torch.empty(0).to(args.device)
val_pred = torch.empty(0).to(args.device)
val_probs = torch.empty(0).to(args.device)
val_logits = torch.empty(0).to(args.device)

model_rgb = architectures.FGN_RGB(args)
model_flow = architectures.FGN_FLOW(args)
model_merge_classify = architectures.FGN_MERGE_CLASSIFY(args)
model = architectures.FGN(model_rgb, model_flow, model_merge_classify)
model = nn.DataParallel(model).to(args.device)

model.load_state_dict(torch.load(os.path.join(paths.models, args.model_name+".pt"), map_location=torch.device('cpu')))
model.eval()
with torch.no_grad():
    if (args.mixed_precision): scaler = torch.cuda.amp.GradScaler() 
    sigmoid = nn.Sigmoid()
    
    for step, s in enumerate(val_loader):
        data_rgb = s['rgb'].to(args.device)
        data_flow = s['flow'].to(args.device)
        label = s['label'].to(args.device)           
        batch_size = data_rgb.size()[0]
        
        if (args.mixed_precision):
            with torch.cuda.amp.autocast():
                logits = torch.squeeze(model(data_rgb, data_flow))
        else:
            logits = torch.squeeze(model(data_rgb, data_flow))
        
        preds = (logits>0).float()
        probs = sigmoid(logits)
        val_true = torch.cat((val_true, label))                        
        val_pred = torch.cat((val_pred, preds))
        val_probs = torch.cat((val_probs, probs))
        val_logits = torch.cat((val_logits, logits))
    
    val_true = val_true.cpu().numpy().astype(int)
    val_pred = val_pred.detach().cpu().numpy().astype(int)
    val_probs = val_probs.detach().cpu().numpy()

    val_acc = metrics.accuracy_score(val_true, val_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(val_true, val_pred)
    val_f1 = metrics.f1_score(val_true, val_pred, average='macro')
    val_precision = metrics.precision_score(val_true, val_pred, average='macro')
    val_recall = metrics.recall_score(val_true, val_pred, average='macro')
    outstr = 'Validation :: val acc: %.6f, val bal acc: %.6f \n val f1 score: %.6f, val precision: %.6f, val recall: %.6f' % (
                                                                                val_acc,
                                                                                avg_per_class_acc,
                                                                                val_f1,
                                                                                val_precision,
                                                                                val_recall)    
    print (outstr)
    utils.plot_confusion_matrix (args, paths, val_true, val_pred, save=True)
    utils.plot_roc_curve (args, paths, val_true, val_probs, save=True)
    
