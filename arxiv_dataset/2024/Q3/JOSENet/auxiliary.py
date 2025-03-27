import argparse
import cv2
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets

import sklearn.metrics as metrics

from torch.utils.data import Dataset, DataLoader
import albumentations as A
import os
import random
from collections import OrderedDict
import yaml


import utils
import preprocessing as pre
import architectures
from roi import *


def parse_args():
    parser = argparse.ArgumentParser(description='roi demo') 

    parser.add_argument('--model_name', default='model_ssl', type=str, help='name of the model/experiment') 
    parser.add_argument('--task', default="VICReg", type=str, help='ssl task: ["O3D", "AoT", "CubicPuzzle", "VICRegSiamese", "VICRegSiameseFlow", "VICReg"]') 
    parser.add_argument('--dataset', default="UCFCrime", type=str, help='pretraining dataset ["HMDB51", "UCF101", "UCFCrime"]') 
    parser.add_argument('--clip_frames', default=16, type=int, help='number of frames per segment (Set 128 for "CubicPuzzle", 32 for "AoT" and 16 for "O3D")') 
    parser.add_argument('--eval', action=argparse.BooleanOptionalAction)

    parsed_args = parser.parse_args()
        
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
args.task                    = parsed_args.task
args.dataset                 = parsed_args.dataset
args.clip_frames             = parsed_args.clip_frames
args.eval                    = parsed_args.eval
args.device                  = device


"""
class Set_Parameters():
    model_name              = parsed_args.model_name
    task                    = parsed_args.task 
    dataset                 = parsed_args.dataset 
    eval                    = parsed_args.eval
    mixed_precision         = True
    clip_frames             = parsed_args.clip_frames 
    interval                = 1       # Used for UCF-Crime
    fps                     = 7.5
    expander_dimensionality = 8192    # 8192    # Used for VICReg
    expander_input_dim      = 864
    batch_size              = 64
    lr                      = 0.02
    eta_min                 = 0.002
    dropout                 = 0.2
    dropout3d               = 0
    epochs                  = 100
    patience                = 100
    
    t_max                   = 100 # T_max: maximum number of iterations. Parameter of the CosineAnnealing
    
    lambd                   = 25
    mu                      = 25
    nu                      = 1
    
    device                  = device
args = Set_Parameters()
"""

class Paths():
    main_folder         = "main"
    jpg_frames          = "ucf_crime_jpg_frames"
    models              = "models/auxiliary" #
paths = Paths()




n_of_frames = args.clip_frames+1 if (args.task == "AoT") else args.clip_frames

if (args.dataset == "HMDB51"):
    
    # fps=None for CubicPuzzle!!!
    # USUALLY: step_between_clips = n_of_frames
    hmdb51_train = datasets.HMDB51('datasets/hmdb51/video_data', 'datasets/hmdb51/test_train_splits/', n_of_frames, frame_rate=args.fps,
                                                    step_between_clips = n_of_frames, fold=1, train=True,
                                                    num_workers=8)
    hmdb51_test = datasets.HMDB51('datasets/hmdb51/video_data', 'datasets/hmdb51/test_train_splits/', n_of_frames, frame_rate=args.fps,
                                                    step_between_clips = n_of_frames, fold=1, train=False,
                                                    num_workers=8)
    x_train = hmdb51_train
    x_val = hmdb51_test
    y_train = []
    y_val = []
    
elif(args.dataset == "UCF101"):
    
    # fps=None for CubicPuzzle!!!
    ucf101_train = datasets.UCF101('datasets/UCF-101/video_data', 'datasets/UCF-101/ucfTrainTestlist/', n_of_frames, frame_rate=args.fps,
                                                    step_between_clips = n_of_frames, fold=1, train=True,
                                                    num_workers=8)
    ucf101_test = datasets.UCF101('datasets/UCF-101/video_data', 'datasets/UCF-101/ucfTrainTestlist/', n_of_frames, frame_rate=args.fps,
                                                    step_between_clips = n_of_frames, fold=1, train=False,
                                                    num_workers=8)
    x_train = ucf101_train
    x_val = ucf101_test
    y_train = []
    y_val = []
    
elif (args.dataset == "UCFCrime"):
    segments = pre.create_window_segments_and_labels_UCFCrime (paths, args)
    x_train = segments['train']
    y_train = []
    x_val = segments['val']
    y_val = []


def generate_subset (data, percentage):
    np.random.seed(8)
    idxs = np.random.randint(0, len(data), int(len(data)*percentage))
    return idxs


if ("VICReg" in args.task):
    if (args.dataset == "UCF101"):
        idxs = generate_subset(x_train, 0.15)
        x_train = torch.utils.data.Subset(x_train, idxs)
        idxs = generate_subset(x_val, 0.15)
        x_val = torch.utils.data.Subset(x_val, idxs)


print (len(x_train))
print (len(x_val))


def odd_one_out (segment, n_questions=6):   
    questions = []
    y = np.random.randint(0, n_questions)
    for i in range(n_questions):
        if (i == y):
            shuffled = np.random.permutation(segment)     
            questions.append(shuffled)
        else: 
            questions.append(segment)
            
    return questions, y

def arrow_of_time (segment):   
    questions = []
    y = np.random.randint(0, 2)
    # Backward the video
    if (y == 0):
        segment = np.flip(segment)

    questions.append(segment[:16])
    questions.append(segment[16:])

    return questions, y



def cubic_puzzle (segment, permutations):   
    questions = []
    y = np.random.randint(len(permutations))
    if (permutations[y]['flip'] == 1):
        #flip the frames upside-down
        segment = np.fliplr(segment)

    #print (np.array(segment).shape) (128, 240, 320, 3)
    # Generation of cuboids 80x80x16
    # From [224, 224, 128] to 4x4 cuboids [112, 112, 32] 
    # From [112, 112, 32] to [80,80,16] with random shift in all directions

    # We obtain this by take as minimum coordinate a random number between [0, 32) and the maximum coordinate between [80, 112)
    # The same reasoning holds for time coordinate "z": min_z = [0,16), max_z = [16,32)  
    x_start = np.random.randint(0, 32)
    y_start = np.random.randint(0, 32)
    z_start = np.random.randint(0,16)
    x_end = x_start + 80
    y_end = y_start + 80
    z_end = z_start + 16
    # Generate 2x2 cuboids for each of the 4 time chunks 
    time_cuboids = [] 
    for t in range(4):
        space_cuboids = []
        '''
        space_cuboids.append(segment[z_start+32*t:z_end+32*t, x_start:x_end,y_start:y_end])
        space_cuboids.append(segment[z_start+32*t:z_end+32*t, x_start+112:x_end+112,y_start:y_end])
        space_cuboids.append(segment[z_start+32*t:z_end+32*t, x_start:x_end,y_start+112:y_end+112])
        space_cuboids.append(segment[z_start+32*t:z_end+32*t,x_start+112:x_end+112,y_start+112:y_end+112])
        '''
        space_cuboids.append(segment[16*t:16*(t+1), x_start:x_end,y_start:y_end])
        space_cuboids.append(segment[16*t:16*(t+1), x_start+112:x_end+112,y_start:y_end])
        space_cuboids.append(segment[16*t:16*(t+1), x_start:x_end,y_start+112:y_end+112])
        space_cuboids.append(segment[16*t:16*(t+1),x_start+112:x_end+112,y_start+112:y_end+112])
        
        time_cuboids.append(space_cuboids)
    #print (np.array(time_cuboids).shape) #(4,4)
    #print (np.array(time_cuboids[0]).shape) #(4,)
    #print (np.array(time_cuboids[0][0]).shape) #(16, 80, 80, 3)
    
    space_time = np.random.rand()
    random_index = np.random.randint(0,4)
    order = permutations[y]["perm"]
    new_cuboids = []
    '''
    for idx in order:
        new_cuboids.append(time_cuboids[random_index][idx])  # rearrange based on permutations
    '''
    # Change the order in space
    if ( space_time < 0.5 ):
        for idx in order:
            new_cuboids.append(time_cuboids[random_index][idx])  # rearrange based on permutations
        '''visualize (new_cuboids[0][0])
        visualize (new_cuboids[1][0])
        visualize (new_cuboids[2][0])
        visualize (new_cuboids[3][0])'''
    #Change the order in time 
    else:
        for idx in order:
            new_cuboids.append(time_cuboids[idx][random_index])
        '''visualize(new_cuboids[0][0])
        visualize(new_cuboids[1][0])
        visualize(new_cuboids[2][0])
        visualize(new_cuboids[3][0])'''
            
    return new_cuboids, y

import itertools
a_list = [0, 1, 2, 3]

permutations_object = itertools.permutations(a_list)
permutations = {}
i = 0
for _,v in enumerate(permutations_object):
    permutations[i] = {"flip": 0, "perm": v} # non-flipped
    i+=1
    '''permutations[i+1] = {"flip": 1, "perm": v} # flipped upside-down
    i+=2 '''


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    if (std == 0):
        return data-mean
    return (data-mean) / std



####################### ODD ONE OUT #########################
class OddOneOut():        
    class Preprocess():    
        resize = A.Compose([
                        A.Resize(224,224)
                        ])
    class Augmentation():    
        rgb = A.ReplayCompose([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5)
                    ]
        )
        rgb_and_flow = A.ReplayCompose([
                    A.HorizontalFlip(p=0.5)
                    ]
        )

    class Create_Dataset(Dataset):
        def __init__(self, videos, labels, args, preprocess, augm=None):

            self.videos = videos
            self.num_samples = len(self.videos)
            self.args = args
            self.augm = augm
            self.preprocess = preprocess


        def __len__(self):
            return self.num_samples    

        def __getitem__(self, idx):

            segment = [] # This is a generic segment, it can be RGB or FLOW depending on "args.SS_input"

            # We took the [0] because the HMDB51 dataset is made of (video, label) but we don't care about label in SS learning
            if (args.dataset == "UCFCrime"):
                rgb_segment = []
                for i,name in enumerate(self.videos[idx]):
                    with open(self.videos[idx][i],'rb') as f: 
                        frame = cv2.imdecode(np.frombuffer(f.read(),dtype=np.uint8), -1)
                        rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
                        rgb_segment.append(rgb)
                segment = [self.preprocess.resize(image=value)['image'] for value in rgb_segment]
            elif (args.dataset == "HMDB51" or args.dataset == "UCF101"):
                rgb_segment = self.videos[idx][0]
                # Apply preprocess to the rgb images
                segment = [self.preprocess.resize(image=value.numpy())['image'] for value in rgb_segment]

            if (self.augm):
                augmented = []
                for i,frame in enumerate(segment):
                    if (i == 0):
                        augmented_frame = self.augm.rgb(image=frame)
                        augmented.append(augmented_frame['image'])
                    else:
                        augmented.append(A.ReplayCompose.replay(augmented_frame['replay'], image=frame)['image'])
                
                segment = augmented
                
                augmented = []
                for i,frame in enumerate(segment):
                    if (i == 0):
                        augmented_frame = self.augm.rgb_and_flow(image=frame)
                        augmented.append(augmented_frame['image'])
                    else:
                        augmented.append(A.ReplayCompose.replay(augmented_frame['replay'], image=frame)['image'])
                
                segment = augmented
            

            # Apply standardization
            segment = normalize(segment)

            questions, y = odd_one_out (segment) 

            # From list to Numpy and Tensor
            questions = [torch.FloatTensor(np.transpose(np.array(segment), (3,0,1,2))) for segment in questions]

            y = torch.tensor(np.array(y)).float()

            item = {'questions': questions,
                    'label': y}   
            return item
        
####################### ARROW OF TIME #########################
class ArrowOfTime ():
    class Preprocess():    
        resize = A.Compose([
                        A.Resize(224,224)
                        ])
    class Augmentation():  
        rgb_and_flow = A.ReplayCompose([
                    A.HorizontalFlip(p=0.5)]                  
        )
    class Create_Dataset(Dataset):
        def __init__(self, videos, labels, args, preprocess, augm=None):

            self.videos = videos
            self.num_samples = len(self.videos)
            self.args = args
            self.augm = augm
            self.preprocess = preprocess


        def __len__(self):
            return self.num_samples    

        def __getitem__(self, idx):

            segment = []

            # We took the [0] because the HMDB51 dataset is made of (video, label) but we don't care about label in SS learning
            if (args.dataset == "UCFCrime"):
                rgb_segment = []
                for i,name in enumerate(self.videos[idx]):
                    with open(self.videos[idx][i],'rb') as f: 
                        frame = cv2.imdecode(np.frombuffer(f.read(),dtype=np.uint8), -1)
                        rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
                        rgb_segment.append(rgb)
                segment = [self.preprocess.resize(image=value)['image'] for value in rgb_segment]
            elif (args.dataset == "HMDB51" or args.dataset == "UCF101"):
                rgb_segment = self.videos[idx][0]
                # Apply preprocess to the rgb images
                segment = [preprocess.resize(image=value.numpy())['image'] for value in rgb_segment]

            flow_segment = []
            for i,frame in enumerate(segment):
                if (i == 0):
                    prev = frame
                else:
                    nexT = frame
                    flow = pre.generate_flow (prev, nexT)
                    flow_segment.append(flow)
                    prev = nexT
            # We do not append last zero frame because otherwise the network will predict too simply the forward/backward task
            segment = flow_segment

            if (self.augm):
                augmented = []
                for i,frame in enumerate(segment):
                    if (i == 0):
                        augmented_frame = self.augm.rgb_and_flow(image=frame)
                        augmented.append(augmented_frame['image'])
                    else:
                        augmented.append(A.ReplayCompose.replay(augmented_frame['replay'], image=frame)['image'])
                
                segment = augmented
            segment = normalize(segment)
            questions, y = arrow_of_time (segment) 
            # From list to Numpy and Tensor
            questions = [torch.FloatTensor(np.transpose(np.array(segment), (3,0,1,2))) for segment in questions]

            y = torch.tensor(np.array(y)).float()
            item = {'questions': questions,
                    'label': y}   
            return item
        
####################### SPACE TIME CUBIC PUZZLE #########################
class CubicPuzzle():        
    class Preprocess():    
        resize = A.Compose([
                        A.Resize(224,224)
                        ])
        def channel_replication(self, segment):
                new_segment = []
                channel_to_replicate = np.random.randint(0,3)
                for frame in segment:
                    (R, G, B) = cv2.split(frame)
                    if (channel_to_replicate == 0):
                        merged = cv2.merge([R, R, R])
                    elif (channel_to_replicate == 1):
                        merged = cv2.merge([G, G, G])
                    else:
                        merged = cv2.merge([B, B, B])
                    new_segment.append(merged)
                return new_segment
    
    class Create_Dataset(Dataset):
        def __init__(self, videos, labels, args, preprocess, augm=None):

            self.videos = videos
            self.num_samples = len(self.videos)
            self.args = args
            self.augm = augm
            self.preprocess = preprocess


        def __len__(self):
            return self.num_samples    

        def __getitem__(self, idx):

            segment = [] # This is a generic segment, it can be RGB or FLOW depending on "args.SS_input"

            # We took the [0] because the HMDB51 dataset is made of (video, label) but we don't care about label in SS learning
            if (args.dataset == "UCFCrime"):
                rgb_segment = []
                for i,name in enumerate(self.videos[idx]):
                    with open(self.videos[idx][i],'rb') as f: 
                        frame = cv2.imdecode(np.frombuffer(f.read(),dtype=np.uint8), -1)
                        rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
                        rgb_segment.append(rgb)
                segment = [self.preprocess.resize(image=value)['image'] for value in rgb_segment]
            elif (args.dataset == "HMDB51" or args.dataset == "UCF101"):
                rgb_segment = self.videos[idx][0]
                # Apply preprocess to the rgb images
                segment = [self.preprocess.resize(image=value.numpy())['image'] for value in rgb_segment]
                        
            if (self.augm):
                # Apply channel replication
                segment = preprocess.channel_replication(segment)
            
            # Apply standardization per segment (because channel replication) 
            segment = normalize(segment)

            questions, y = cubic_puzzle(np.array(segment), permutations)
            # Now we have 4 questions made of 16x80x80 segments            
            
            # From list to Numpy and Tensor
            questions = [torch.FloatTensor(np.transpose(np.array(segment), (3,0,1,2))) for segment in questions]
            y = torch.tensor(np.array(y)).float()

            item = {'questions': questions,
                    'label': y}   
            return item
        


####################### VICRegSiamese #########################
class VICRegSiamese():   
    class Preprocess():
        pass
    class Augmentation():   
        rgb = A.ReplayCompose([                 
                    A.RandomResizedCrop(224, 224, scale=(0.08, 1), interpolation=cv2.INTER_CUBIC),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, always_apply=False, p=0.8),
                    A.ToGray(p = 0.2),
                    A.GaussianBlur(blur_limit=[23, 23], p=0.5), # different in the implementation
                    A.Solarize(p = 0.1)                         # different in the implementation      
            ])     

    
    class Create_Dataset(Dataset):
        def __init__(self, videos, labels, args, preprocess, augm=None):

            self.videos = videos
            self.num_samples = len(self.videos)
            self.args = args
            self.augm = augm
            self.preprocess = preprocess


        def __len__(self):
            return self.num_samples    

        def __getitem__(self, idx):

            segment = [] # This is a generic segment, it can be RGB or FLOW depending on "args.SS_input"

            # We took the [0] because the HMDB51 dataset is made of (video, label) but we don't care about label in SS learning
            if (args.dataset == "UCFCrime"):
                rgb_segment = []
                for i,name in enumerate(self.videos[idx]):
                    with open(self.videos[idx][i],'rb') as f: 
                        frame = cv2.imdecode(np.frombuffer(f.read(),dtype=np.uint8), -1)
                        rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
                        rgb_segment.append(rgb)
                #segment = [self.preprocess.resize(image=value)['image'] for value in rgb_segment]
            elif (args.dataset == "HMDB51" or args.dataset == "UCF101"):
                rgb_segment = self.videos[idx][0]
                # Apply preprocess to the rgb images
                segment = [value.numpy() for value in rgb_segment]
                #segment = [self.preprocess.resize(image=value.numpy())['image'] for value in rgb_segment]
            # Create two different augmented version of the videos
            questions = []
            for i in range(2):
                augmented = []
                for i,frame in enumerate(segment):
                    if (i == 0):
                        augmented_frame = self.augm.rgb(image=frame)
                        augmented.append(augmented_frame['image'])
                    else:
                        augmented.append(A.ReplayCompose.replay(augmented_frame['replay'], image=frame)['image'])

                # Apply standardization per segment and append to the list of questions (len = 2)
                questions.append(normalize(augmented))
            
            #visualize(questions[0][0])
            #visualize(questions[1][0])
            
            # From list to Numpy and Tensor
            questions = [torch.FloatTensor(np.transpose(np.array(segment), (3,0,1,2))) for segment in questions]
            y = 0 # Set to zero

            item = {'questions': questions,
                    'label': y}   
            return item
        



####################### VICRegSiameseFlow #########################
class VICRegSiameseFlow():   
    class Preprocess():
        pass
    class Augmentation():      
        flow = A.ReplayCompose([
                #A.Resize(224,224),
                A.RandomResizedCrop(224, 224, scale=(0.08, 1), interpolation=cv2.INTER_CUBIC),
                A.HorizontalFlip(p=0.5)
            ]) 

    
    class Create_Dataset(Dataset):
        def __init__(self, videos, labels, args, preprocess, augm=None):

            self.videos = videos
            self.num_samples = len(self.videos)
            self.args = args
            self.augm = augm
            self.preprocess = preprocess


        def __len__(self):
            return self.num_samples    

        def __getitem__(self, idx):

            segment = [] # This is a generic segment, it can be RGB or FLOW depending on "args.SS_input"

            # We took the [0] because the HMDB51 dataset is made of (video, label) but we don't care about label in SS learning
            if (args.dataset == "UCFCrime"):
                rgb_segment = []
                for i,name in enumerate(self.videos[idx]):
                    with open(self.videos[idx][i],'rb') as f: 
                        frame = cv2.imdecode(np.frombuffer(f.read(),dtype=np.uint8), -1)
                        rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
                        rgb_segment.append(rgb)
                #segment = [self.preprocess.resize(image=value)['image'] for value in rgb_segment]
            elif (args.dataset == "HMDB51" or args.dataset == "UCF101"):
                rgb_segment = self.videos[idx][0]
                # Apply preprocess to the rgb images
                segment = [value.numpy() for value in rgb_segment]
                #segment = [self.preprocess.resize(image=value.numpy())['image'] for value in rgb_segment]
                
            flow_segment = []
            for i,frame in enumerate(segment):
                if (i == 0):
                    prev = frame
                else:
                    nexT = frame
                    flow = pre.generate_flow (prev, nexT)
                    flow_segment.append(flow)
                    prev = nexT
            flow_segment.append(np.zeros(( flow.shape[0],flow.shape[1],flow.shape[2])))
                    
            # Create two different augmented version of the videos
            questions = []
            for i in range(2):
                augmented = []
                for i,frame in enumerate(flow_segment):
                    if (i == 0):
                        augmented_frame = self.augm.flow(image=frame)
                        augmented.append(augmented_frame['image'])
                    else:
                        augmented.append(A.ReplayCompose.replay(augmented_frame['replay'], image=frame)['image'])
                
                questions.append(normalize(augmented))
            
            
            '''utils.visualize(questions[0][14][:,:,0])
            utils.visualize(questions[1][14][:,:,0])'''
            # From list to Numpy and Tensor
            questions = [torch.FloatTensor(np.transpose(np.array(segment), (3,0,1,2))) for segment in questions]
            y = 0 # Set to zero

            item = {'questions': questions,
                    'label': y}   
            return item
        


####################### VICReg #########################
class VICReg():   
    class Preprocess():
        resize = A.Compose([
                        A.Resize(224,224)
                        ])
    class Augmentation():   
        rgb = A.ReplayCompose([      
                    A.RandomResizedCrop(112, 112, scale=(0.08, 0.1), interpolation=cv2.INTER_CUBIC),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, always_apply=False, p=0.8),
                    A.ToGray(p = 0.2),
                    A.GaussianBlur(blur_limit=[23, 23], p=0.5), # different in the implementation
                    A.Solarize(p = 0.1)                         # different in the implementation
                    
            ])
        flow = A.ReplayCompose([
                    A.Resize(112,112),
                    A.HorizontalFlip(p=0.5)
            ]) 

    
    class Create_Dataset(Dataset):
        def __init__(self, videos, labels, args, preprocess, augm):

            self.videos = videos
            self.num_samples = len(self.videos)
            self.args = args
            self.preprocess = preprocess
            self.augm = augm



        def __len__(self):
            return self.num_samples    

        def __getitem__(self, idx):

            segment = []

            # We took the [0] because the HMDB51 dataset is made of (video, label) but we don't care about label in SS learning
            if (args.dataset == "UCFCrime"):
                rgb_segment = []
                for i,name in enumerate(self.videos[idx]):
                    with open(self.videos[idx][i],'rb') as f: 
                        frame = cv2.imdecode(np.frombuffer(f.read(),dtype=np.uint8), -1)
                        rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
                        rgb_segment.append(rgb)
                #segment = [self.preprocess.resize(image=value)['image'] for value in rgb_segment]
            elif (args.dataset == "HMDB51" or args.dataset == "UCF101"):
                rgb_segment = self.videos[idx][0]
                # Apply preprocess to the rgb images
                segment = [value.numpy() for value in rgb_segment]
                #segment = [self.preprocess.resize(image=value.numpy())['image'] for value in rgb_segment]
            
            '''##### ADDED
            idx_rgb = np.random.choice(16, size=4, replace=False)
            idx_flow = np.random.choice(15,size=3, replace=False)
            
            flow_segment = []
            for i,frame in enumerate(segment):
                if (i == 0):
                    prev = frame
                    if (i in idx_rgb):
                        segment[i] = np.zeros((frame.shape[0],frame.shape[1],frame.shape[2]), dtype=np.uint8)
                        
                else:
                    if (i-1 in idx_flow):
                        flow_segment.append(np.zeros((frame.shape[0],frame.shape[1], 2)))
                    else:
                        nexT = frame
                        flow = pre.generate_flow (prev, nexT)
                        flow_segment.append(flow)
                        prev = nexT
                        
                    if (i in idx_rgb):
                        segment[i] = np.zeros((frame.shape[0],frame.shape[1],frame.shape[2]), dtype=np.uint8)
                        
                                              
            flow_segment.append(np.zeros((frame.shape[0],frame.shape[1], 2)))
            ######'''
            
            flow_segment = []
            for i,frame in enumerate(segment):
                if (i == 0):
                    prev = frame
                else:
                    nexT = frame
                    flow = pre.generate_flow (prev, nexT)
                    flow_segment.append(flow)
                    prev = nexT
                    
            flow_segment.append(np.zeros((flow.shape[0],flow.shape[1],flow.shape[2])))
            
            '''
            # ADDED
            segment = [self.preprocess.resize(image=value)['image'] for value in segment]
            flow_segment = [self.preprocess.resize(image=value)['image'] for value in flow_segment]
            segment = roi_video(segment, flow_segment, seed=False)
            ########
            '''
            
            ##### Create two different inputs
            questions = []
            
            if (self.augm):
                # Augment the RGB segment
                augmented = []
                for i,frame in enumerate(segment):
                    if (i == 0):
                        augmented_frame = self.augm.rgb(image=frame)
                        augmented.append(augmented_frame['image'])
                    else:
                        augmented.append(A.ReplayCompose.replay(augmented_frame['replay'], image=frame)['image'])

                # Apply standardization per segment and append to the list of questions (len = 2)
                
                questions.append(normalize(augmented))
                
                
                # Augment the FLOW segment
                augmented = []
                for i,frame in enumerate(flow_segment):
                    if (i == 0):
                        augmented_frame = self.augm.flow(image=frame)
                        augmented.append(augmented_frame['image'])
                    else:
                        augmented.append(A.ReplayCompose.replay(augmented_frame['replay'], image=frame)['image'])

                # Apply standardization per segment and append to the list of questions (len = 2)
                questions.append(normalize(augmented))
            else:
                segment = [self.preprocess.resize(image=value)['image'] for value in segment]
                flow_segment = [self.preprocess.resize(image=value)['image'] for value in flow_segment]
                questions.append(normalize(segment))
                questions.append(normalize(flow_segment))
            
            '''utils.visualize(questions[0][0])
            utils.visualize(questions[0][10])
            utils.visualize(questions[1][0][:,:,0])
            utils.visualize(questions[1][10][:,:,0])'''
            # From list to Numpy and Tensor
            questions = [torch.FloatTensor(np.transpose(np.array(segment), (3,0,1,2))) for segment in questions]
            y = self.videos[idx][2]
            
            item = {'questions': questions,
                    'label': y}   
            return item
        

if (args.task == "O3D"):
    task = OddOneOut()
elif (args.task == "AoT"):
    task = ArrowOfTime()
elif (args.task == "CubicPuzzle"):
    task = CubicPuzzle()
elif (args.task == "VICRegSiamese"):
    task = VICRegSiamese()
elif (args.task == "VICRegSiameseFlow"):
    task = VICRegSiameseFlow()
elif (args.task == "VICReg"):
    task = VICReg()
preprocess = task.Preprocess()    
augm = task.Augmentation()    

train_loader = DataLoader(task.Create_Dataset(x_train, y_train, args, preprocess, augm=augm),
                                batch_size=args.batch_size, num_workers = 8, shuffle=True, drop_last=True)

val_loader = DataLoader(task.Create_Dataset(x_val, y_val, args, preprocess, augm=None),
                            batch_size=args.batch_size, num_workers = 8, shuffle=True, drop_last=True)


#task.Create_Dataset(x_val, y_val, args, preprocess, augm=augm).__getitem__(140)['questions'][0][0][0]
task.Create_Dataset(x_val, y_val, args, preprocess, augm=augm).__getitem__(0)['label']


if (args.task == "O3D"):
    model_rgb = architectures.FGN_RGB(args)
    model = architectures.Odd_One_Out(args, model_rgb)

elif (args.task == "AoT"):
    model_flow = architectures.FGN_FLOW(args)
    model = architectures.Arrow_Of_Time(model_flow)
    
elif (args.task == "CubicPuzzle"):
    model_rgb = architectures.FGN_RGB(args)
    model = architectures.Cubic_Puzzle(model_rgb)
    
elif (args.task == "VICRegSiamese"):
    model_rgb = architectures.FGN_RGB(args)
    model = architectures.VIC_Reg_Siamese(args, model_rgb)
    
elif (args.task == "VICRegSiameseFlow"):
    model_flow = architectures.FGN_FLOW(args)
    model = architectures.VIC_Reg_Siamese(args, model_flow)

elif (args.task == "VICReg"):
    model_rgb = architectures.FGN_RGB(args)
    model_flow = architectures.FGN_FLOW(args)
    merging_block = architectures.MergingBlock()
    model = architectures.VIC_Reg(args, model_rgb, model_flow, merging_block)
           
model = model.to(args.device)
#summary(model, (args.batch_size, 3, 16, 224, 224))



if (not args.eval and not "VICReg" in args.task):
    #opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = CosineAnnealingLR(opt, args.t_max, eta_min=args.eta_min)
    
    train_loss_tot = []
    valid_loss_tot = []
    
    # Early stopping
    best_val_loss, best_val_epoch = None, None
    best_metric = 0
    for epoch in tqdm(range(args.epochs)):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true = torch.empty(0).to(args.device)
        train_pred = torch.empty(0).to(args.device)
        
        if (args.mixed_precision): scaler = torch.cuda.amp.GradScaler() 
            
        for step, s in enumerate(train_loader):              
            data_rgb = [s.to(args.device) for s in s['questions']]
            label = s['label'].to(args.device)           
            
            batch_size = args.batch_size
            opt.zero_grad()
            
            if (args.mixed_precision):
                with torch.cuda.amp.autocast():
                    logits = torch.squeeze(model(data_rgb)) 
                    label = label.to(torch.int64)

                    loss = F.cross_entropy(logits, label)
                    #loss = focal_loss(logits, label)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = torch.squeeze(model(data_rgb)) 
                label = label.to(torch.int64)

                loss = F.cross_entropy(logits, label)
                #loss = focal_loss(logits, label)
                loss.backward()
                opt.step()
            
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            
            train_true = torch.cat((train_true, label))                        
            train_pred = torch.cat((train_pred, preds))
            #torch.cuda.empty_cache()
            #if (step == 100): break
        
    
        
        train_loss = train_loss*1.0/count
        train_loss_tot.append(train_loss)  
        train_true = train_true.cpu().numpy().astype(int)
        train_pred = train_pred.detach().cpu().numpy().astype(int)
        
        if (epoch == 0): utils.output_histogram(train_pred, args.n_outputs) 
        
        #train_true = np.concatenate(train_true)
        #train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        train_f1 = metrics.f1_score(train_true, train_pred, average='macro')
        train_precision = metrics.precision_score(train_true, train_pred, average='macro')
        train_recall = metrics.recall_score(train_true, train_pred, average='macro')
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
        #val_pred = []
        with torch.no_grad():
            
            if (args.mixed_precision): scaler = torch.cuda.amp.GradScaler() 
            
            for step, s in enumerate(val_loader):
                data_rgb = [s.to(args.device) for s in s['questions']]
                #data_flow = s['flow'].to(args.device)
                label = s['label'].to(args.device)           
            
                batch_size = args.batch_size
                opt.zero_grad()
            
                if (args.mixed_precision):
                    with torch.cuda.amp.autocast():
                        logits = torch.squeeze(model(data_rgb)) 
                        label = label.to(torch.int64)

                        loss = F.cross_entropy(logits, label)
                        #loss = focal_loss(logits, label)
                else:
                    logits = torch.squeeze(model(data_rgb)) 
                    label = label.to(torch.int64)

                    loss = F.cross_entropy(logits, label)
                    #loss = focal_loss(logits, label)

                preds = logits.max(dim=1)[1]
                count += batch_size
                val_loss += loss.item() * batch_size
                val_true = torch.cat((val_true, label))                        
                val_pred = torch.cat((val_pred, preds))
                #torch.cuda.empty_cache()
                #if (step == 10): break
            scheduler.step()     
            
            val_loss = val_loss*1.0/count
            valid_loss_tot.append(val_loss)
            val_true = val_true.cpu().numpy().astype(int)
            val_pred = val_pred.detach().cpu().numpy().astype(int)
            #val_true = np.concatenate(val_true)
            #val_pred = np.concatenate(val_pred)
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
            metric = val_f1
            if metric >= best_metric:
                best_metric = metric
                if not os.path.exists(paths.models): os.makedirs(paths.models)
                torch.save(model.state_dict(), os.path.join(paths.models, args.model_name+".pt"))
                print('----- Model saved -----')
            if (best_val_loss is None or val_loss < best_val_loss):
                best_val_loss, best_val_epoch = val_loss, epoch
                print ("----- Validation Loss decreased! -----")
            '''
            if best_val_epoch < epoch + 1 - args.patience:
                # nothing is improving for a while
                print ("----- Early stopping -----")
                break  
            '''
            
    utils.plot_loss_function(args, paths, train_loss_tot, valid_loss_tot, epoch)
    print('--- End ---')
else:
    print ("----- Evaluation -----")



if (not "VICReg" in args.task):
    test_acc = 0.0
    count = 0.0
    val_true = torch.empty(0).to(args.device)
    val_pred = torch.empty(0).to(args.device)

    model.load_state_dict(torch.load(os.path.join(paths.models, args.model_name+".pt")))
    model.eval()
    with torch.no_grad():
        if (args.mixed_precision): scaler = torch.cuda.amp.GradScaler() 
        sigmoid = nn.Sigmoid()
        #torch.cuda.synchronize()
        #since = int(round(time.time()*1000))

        for step, s in enumerate(val_loader):
            data_rgb = [s.to(args.device) for s in s['questions']]
            label = s['label'].to(args.device)    
            label = s['label'].to(args.device)           
            batch_size = args.batch_size

            if (args.mixed_precision):
                with torch.cuda.amp.autocast():
                    logits = torch.squeeze(model(data_rgb))
            else:
                logits = torch.squeeze(model(data_rgb))

            preds = logits.max(dim=1)[1]
            val_true = torch.cat((val_true, label))                        
            val_pred = torch.cat((val_pred, preds))

        #torch.cuda.synchronize()
        #time_elapsed = (int(round(time.time()*1000)) - since)/1000
        #num_samples = Create_Dataset(x_val, val_labels, args, augm=None).num_samples
        #tot_frames = 16*num_samples
        #fps = tot_frames/time_elapsed
        #print ('time elapsed {}s'.format(time_elapsed))
        #print ('Frame Per Second {}'.format(fps))

        val_true = val_true.cpu().numpy().astype(int)
        val_pred = val_pred.detach().cpu().numpy().astype(int)
        #val_true = np.concatenate(val_true)
        #val_pred = np.concatenate(val_pred)
        val_acc = metrics.accuracy_score(val_true, val_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(val_true, val_pred)
        val_f1 = metrics.f1_score(val_true, val_pred, average='macro')
        val_precision = metrics.precision_score(val_true, val_pred, average='macro')
        val_recall = metrics.recall_score(val_true, val_pred, average='macro')
        outstr = 'val acc: %.6f, val avg acc: %.6f \n val f1 score: %.6f, val precision: %.6f, val recall: %.6f' % (
                                                                                val_acc,
                                                                                avg_per_class_acc,
                                                                                val_f1,
                                                                                val_precision,
                                                                                val_recall)
        print(outstr)


#### Train for VICReg #####
#opt = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-6, amsgrad=True)
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6, nesterov=True)
scheduler = CosineAnnealingLR(opt, args.t_max, eta_min=args.eta_min)


start_epoch = 0
best_metric = 999999
best_train_loss, best_train_epoch = None, None
train_loss_tot = []


if (not args.eval and "VICReg" in args.task):    
    # Early stopping
    for epoch in tqdm(range(start_epoch, args.epochs)):
        
        ####################
        # Train
        ####################
        train_loss = 0.0
        repr_loss, std_loss, cov_loss = 0.0, 0.0, 0.0
        count = 0.0
        
        model.train()
        
        if (args.mixed_precision): scaler = torch.cuda.amp.GradScaler() 
            
        for step, s in enumerate(train_loader):              
            data = [s.to(args.device) for s in s['questions']]         
            
            batch_size = args.batch_size
            opt.zero_grad()
            
            if (args.mixed_precision):
                with torch.cuda.amp.autocast():
                    loss, rloss, sloss, closs = model(data)
                '''
                print (rloss)
                print (sloss)
                print (closs)
                print ("_____")
                '''
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()                
            else:
                loss, repr_loss, std_loss, cov_loss = model(data)
                loss.backward()
                opt.step()
            
            train_loss +=loss.item()
            repr_loss  +=rloss
            std_loss   +=sloss
            cov_loss   +=closs
            count      +=1
            
            #if (step == 100): break
    
        train_loss = train_loss*1.0/count
        train_loss_tot.append(train_loss)  
                
        outstr = '\nTrain %d, loss: %.6f, repr_loss: %.6f, std_loss: %.6f, cov_loss: %.6f' % (epoch,
                                                                                              train_loss,
                                                                                              repr_loss/count,
                                                                                              std_loss/count,
                                                                                              cov_loss/count)
        print(outstr)
                       
        scheduler.step()                                                                    
        
        # Save best model
        metric = train_loss
        if metric <= best_metric:
            best_metric = metric
            #torch.save(model.state_dict(), os.path.join(paths.models, args.model_name+".pt"))
            # Save only the parameters that will be loaded for the training, discard fc layers and batchNorm1d (too large size)
            param_to_save = OrderedDict()
            for name, param in model.named_parameters():
                if ("model" in name or "block" in name):
                    param_to_save [name] = param
            if not os.path.exists(paths.models): os.makedirs(paths.models)
            torch.save(param_to_save, os.path.join(paths.models, args.model_name+".pt"))
            print('----- Model saved -----')
        
        # Early stopping 
        if (best_train_loss is None or train_loss < best_train_loss):
            best_train_loss, best_train_epoch = train_loss, epoch
            print ("----- Loss decreased! -----")
        if best_train_epoch < epoch + 1 - args.patience:
            # nothing is improving for a while
            print ("----- Early stopping -----")
            break  
        
        """
        state = dict(
                epoch=epoch+1,
                model=model.state_dict(),
                opt=opt.state_dict(),
                scheduler=scheduler.state_dict(),
                best_metric=best_metric,
                best_train_epoch = best_train_epoch,
                best_train_loss = best_train_loss,
                train_loss_tot=train_loss_tot
            )
        if not os.path.exists(paths.models): os.makedirs(paths.models)
        torch.save(state, os.path.join(paths.models, args.model_name+'.pth'))
        """
    #utils.plot_loss_function(args, paths, train_loss_tot, valid_loss_tot, epoch-1)
    print('--- End ---')
else:
    print ("----- Evaluation -----")