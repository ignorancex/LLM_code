import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
from dataloaders.helper import CutoutPIL
from randaugment import RandAugment
import pickle, json


class Cholec(data.Dataset):
    def __init__(self, root, data_split, img_size=224, p=1, annFile="", label_mask=None, partial=1+1e-6):
        # data_split = train / val
        self.data_path = "/home/swalimbe/datasets"
        self.data_split = data_split
        self.class_num = 110
        self.classnames = [
            'Preparation',
            'Calot Triangle Dissection',
            'Clipping Cutting',
            'Gallbladder Dissection',
            'Gallbladder Retraction',
            'Cleaning Coagulation',
            'Gallbladder Packaging',
            'Seeing two structures cystic duct and cystic artery',
            'Carefully dissected hepatocystic triangle presenting an unimpeded view of only the 2 cystic structures and the cystic plate',
            'Lower part of the gallbladder divided from the liver bed to expose the cystic plate',
            'The tool grasper performing the action dissect on the target cystic plate',
            'The tool grasper performing the action dissect on the target gallbladder',
            'The tool grasper performing the action dissect on the target omentum',
            'The tool grasper performing the action grasp on the target cystic artery',
            'The tool grasper performing the action grasp on the target cystic duct',
            'The tool grasper performing the action grasp on the target cystic pedicle',
            'The tool grasper performing the action grasp on the target cystic plate',
            'The tool grasper performing the action grasp on the target gallbladder',
            'The tool grasper performing the action grasp on the target gut',
            'The tool grasper performing the action grasp on the target liver',
            'The tool grasper performing the action grasp on the target omentum',
            'The tool grasper performing the action grasp on the target peritoneum',
            'The tool grasper performing the action grasp on the target specimen bag',
            'The tool grasper performing the action pack on the target gallbladder',
            'The tool grasper performing the action retract on the target cystic duct',
            'The tool grasper performing the action retract on the target cystic pedicle',
            'The tool grasper performing the action retract on the target cystic plate',
            'The tool grasper performing the action retract on the target gallbladder',
            'The tool grasper performing the action retract on the target gut',
            'The tool grasper performing the action retract on the target liver',
            'The tool grasper performing the action retract on the target omentum',
            'The tool grasper performing the action retract on the target peritoneum',
            'The tool bipolar performing the action coagulate on the target abdominal wall cavity',
            'The tool bipolar performing the action coagulate on the target blood vessel',
            'The tool bipolar performing the action coagulate on the target cystic artery',
            'The tool bipolar performing the action coagulate on the target cystic duct',
            'The tool bipolar performing the action coagulate on the target cystic pedicle',
            'The tool bipolar performing the action coagulate on the target cystic plate',
            'The tool bipolar performing the action coagulate on the target gallbladder',
            'The tool bipolar performing the action coagulate on the target liver',
            'The tool bipolar performing the action coagulate on the target omentum',
            'The tool bipolar performing the action coagulate on the target peritoneum',
            'The tool bipolar performing the action dissect on the target adhesion',
            'The tool bipolar performing the action dissect on the target cystic artery',
            'The tool bipolar performing the action dissect on the target cystic duct',
            'The tool bipolar performing the action dissect on the target cystic plate',
            'The tool bipolar performing the action dissect on the target gallbladder',
            'The tool bipolar performing the action dissect on the target omentum',
            'The tool bipolar performing the action grasp on the target cystic plate',
            'The tool bipolar performing the action grasp on the target liver',
            'The tool bipolar performing the action grasp on the target specimen bag',
            'The tool bipolar performing the action retract on the target cystic duct',
            'The tool bipolar performing the action retract on the target cystic pedicle',
            'The tool bipolar performing the action retract on the target gallbladder',
            'The tool bipolar performing the action retract on the target liver',
            'The tool bipolar performing the action retract on the target omentum',
            'The tool hook performing the action coagulate on the target blood vessel',
            'The tool hook performing the action coagulate on the target cystic artery',
            'The tool hook performing the action coagulate on the target cystic duct',
            'The tool hook performing the action coagulate on the target cystic pedicle',
            'The tool hook performing the action coagulate on the target cystic plate',
            'The tool hook performing the action coagulate on the target gallbladder',
            'The tool hook performing the action coagulate on the target liver',
            'The tool hook performing the action coagulate on the target omentum',
            'The tool hook performing the action cut on the target blood vessel',
            'The tool hook performing the action cut on the target peritoneum',
            'The tool hook performing the action dissect on the target blood vessel',
            'The tool hook performing the action dissect on the target cystic artery',
            'The tool hook performing the action dissect on the target cystic duct',
            'The tool hook performing the action dissect on the target cystic plate',
            'The tool hook performing the action dissect on the target gallbladder',
            'The tool hook performing the action dissect on the target omentum',
            'The tool hook performing the action dissect on the target peritoneum',
            'The tool hook performing the action retract on the target gallbladder',
            'The tool hook performing the action retract on the target liver',
            'The tool scissors performing the action coagulate on the target omentum',
            'The tool scissors performing the action cut on the target adhesion',
            'The tool scissors performing the action cut on the target blood vessel',
            'The tool scissors performing the action cut on the target cystic artery',
            'The tool scissors performing the action cut on the target cystic duct',
            'The tool scissors performing the action cut on the target cystic plate',
            'The tool scissors performing the action cut on the target liver',
            'The tool scissors performing the action cut on the target omentum',
            'The tool scissors performing the action cut on the target peritoneum',
            'The tool scissors performing the action dissect on the target cystic plate',
            'The tool scissors performing the action dissect on the target gallbladder',
            'The tool scissors performing the action dissect on the target omentum',
            'The tool clipper performing the action clip on the target blood vessel',
            'The tool clipper performing the action clip on the target cystic artery',
            'The tool clipper performing the action clip on the target cystic duct',
            'The tool clipper performing the action clip on the target cystic pedicle',
            'The tool clipper performing the action clip on the target cystic plate',
            'The tool irrigator performing the action aspirate on the target fluid',
            'The tool irrigator performing the action dissect on the target cystic duct',
            'The tool irrigator performing the action dissect on the target cystic pedicle',
            'The tool irrigator performing the action dissect on the target cystic plate',
            'The tool irrigator performing the action dissect on the target gallbladder',
            'The tool irrigator performing the action dissect on the target omentum',
            'The tool irrigator performing the action irrigate on the target abdominal wall cavity',
            'The tool irrigator performing the action irrigate on the target cystic pedicle',
            'The tool irrigator performing the action irrigate on the target liver',
            'The tool irrigator performing the action retract on the target gallbladder',
            'The tool irrigator performing the action retract on the target liver',
            'The tool irrigator performing the action retract on the target omentum',
            'The tool grasper performing the action null verb on the target null target',
            'The tool bipolar performing the action null verb on the target null target',
            'The tool hook performing the action null verb on the target null target',
            'The tool scissors performing the action null verb on the target null target',
            'The tool clipper performing the action null verb on the target null target',
            'The tool irrigator performing the action null verb on the target null target',
        ]

        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(img_size)
            transforms.Resize((img_size, img_size)),
            #CutoutPIL(cutout_factor=0.5),
            #RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        test_transform = transforms.Compose([
            # transforms.CenterCrop(img_size),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        if self.data_split == 'train' or self.data_split == 'val':
            self.transform = train_transform
        elif self.data_split == 'test':
            self.transform = test_transform
        else:
            raise ValueError('data split = %s is not supported in Cholec' % self.data_split)

        self.data = self.read_data()
    
    def __getitem__(self,index):
        impath, label, vid = self.data[index]
        image = Image.open(impath).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.data_split == 'train':
            assert torch.sum(label) < 2
        
        return image, label, vid
    
    def __len__(self):
        return len(self.data)
    
    def random_pick_one(self,tensor):

        ones_indices = torch.nonzero(tensor, as_tuple=False)
        
        if ones_indices.shape[0] > 0:
            random_idx = torch.randint((ones_indices.shape[0]), (1,))
            selected_index = ones_indices[random_idx]
            tensor.zero_()
            tensor[selected_index] = 1

        return tensor
    
    def read_data(self):
            
        if self.data_split == 'val':    

            data = []

            # For cholec80
            cholec80_frames = os.path.join(self.data_path,"cholec80/frames/val")
            cholec80_labels = os.path.join(self.data_path,"cholec80/labels/val/1fps.pickle")
            a = pickle.load(open(cholec80_labels,"rb"))
            invalid_80 = [42]
            for video in a.keys():
                id = int(video[5:])
                if id in invalid_80:
                    continue
                video_folder = os.path.join(cholec80_frames,video)
                input = [(p["Frame_id"],p["Phase_gt"]) for p in a[video]]
                video_id = f"cholec80_{video}"
                for image,gt in input:
                    filename = f"{image}.jpg"
                    impath = os.path.join(video_folder,filename)
                    label = torch.zeros(self.class_num)
                    label[gt] = 1
                    data.append([impath,label,video_id])
            
            # For endoscapes
            def read_json(file_path):
                with open(file_path, 'r') as file:
                    data = json.load(file)
                return data
            
            endo_path = os.path.join(self.data_path,"endoscapes/val")
            b = read_json(os.path.join(endo_path,"annotation_ds_coco.json"))
            for p in b['images']:
                filename,gt,video = (p['file_name'],(p['ds']),p['video_id'])
                impath = os.path.join(endo_path,filename)
                gt = [round(value) for value in gt]
                label = torch.zeros(self.class_num)
                for i in range(len(gt)):
                    if gt[i] == 1:
                        label[i+7] = 1
                video_id = f"endoscapes_{video}"
                data.append([impath,label,video_id])
            
            # For cholect50
            val_split = [8,12,29,50,78]
            cholect50_frames = os.path.join(self.data_path,"cholect50/videos")
            cholect50_labels = os.path.join(self.data_path,"cholect50/labels")

            def get_label(labels):

                output = torch.zeros(self.class_num)
                for label in labels:
                    index = label[0]
                    if index == -1:
                        continue
                    output[index+10] = 1
                    
                return output

            for i in val_split:
                video_folder = os.path.join(cholect50_frames,f"VID{i:02d}" if i < 100 else f"VID{i:03d}")
                label_file = os.path.join(cholect50_labels,f"VID{i:02d}.json" if i < 100 else f"VID{i:03d}.json")
                video_id = f"cholect50_{i}"

                with open(label_file, 'r') as file:
                    a = json.load(file)

                for frame_id,gts in a['annotations'].items():
                    filename = f"{int(frame_id):06d}.png"
                    impath = os.path.join(video_folder,filename)
                    label = get_label(gts)
                    label = torch.Tensor(label)
                    data.append([impath,label,video_id])

            return data
        
        elif self.data_split == 'train':

            data = []

            # For cholec80
            cholec80_frames = os.path.join(self.data_path,"cholec80/frames/train")
            cholec80_labels = os.path.join(self.data_path,"cholec80/labels/train/1fps_100_0.pickle")
            a = pickle.load(open(cholec80_labels,"rb"))
            invalid_80 = [6,10,14,32]
            for video in a.keys():
                id = int(video[5:])
                if id in invalid_80:
                    continue
                video_folder = os.path.join(cholec80_frames,video)
                input = [(p["Frame_id"],p["Phase_gt"]) for p in a[video]]
                video_id = f"cholec80_{video}"
                for image,gt in input:
                    filename = f"{image}.jpg"
                    impath = os.path.join(video_folder,filename)
                    label = torch.zeros(self.class_num)
                    label[gt] = 1
                    data.append([impath,label,video_id])
            
            # For endoscapes
            def read_json(file_path):
                with open(file_path, 'r') as file:
                    data = json.load(file)
                return data
            
            endo_path = os.path.join(self.data_path,"endoscapes/train")
            b = read_json(os.path.join(endo_path,"annotation_ds_coco.json"))
            for p in b['images']:
                filename,gt,video = (p['file_name'],(p['ds']),p['video_id'])
                impath = os.path.join(endo_path,filename)
                gt = [round(value) for value in gt]
                label = torch.zeros(self.class_num)
                for i in range(len(gt)):
                    if gt[i] == 1:
                        label[i+7] = 1
                video_id = f"endoscapes_{video}"
                label = self.random_pick_one(label)
                data.append([impath,label,video_id])
            
            # For cholect50
            train_split = [1,2,4,5,13,15,18,22,23,25,26,27,31,35,36,40,43,47,48,49,52,56,57,60,62,65,66,68,70,75,79,92,96,103,110]
            cholect50_frames = os.path.join(self.data_path,"cholect50/videos")
            cholect50_labels = os.path.join(self.data_path,"cholect50/labels")

            def get_label(labels):

                output = torch.zeros(self.class_num)
                for label in labels:
                    index = label[0]
                    if index == -1:
                        continue
                    output[index+10] = 1
                    
                return output

            for i in train_split:
                video_folder = os.path.join(cholect50_frames,f"VID{i:02d}" if i < 100 else f"VID{i:03d}")
                label_file = os.path.join(cholect50_labels,f"VID{i:02d}.json" if i < 100 else f"VID{i:03d}.json")
                video_id = f"cholect50_{i}"

                with open(label_file, 'r') as file:
                    a = json.load(file)

                for frame_id,gts in a['annotations'].items():
                    filename = f"{int(frame_id):06d}.png"
                    impath = os.path.join(video_folder,filename)
                    label = get_label(gts)
                    label = self.random_pick_one(label)
                    data.append([impath,label,video_id])

            assert len(set([d[0] for d in data])) == len([d[1] for d in data]) == len(data)
            
            return data
    
        elif self.data_split == 'test':

            data = []

            # For cholec80
            cholec80_frames = os.path.join(self.data_path,"cholec80/frames/test")
            cholec80_labels = os.path.join(self.data_path,"cholec80/labels/test/1fps.pickle")
            a = pickle.load(open(cholec80_labels,"rb"))
            valid_80 = [51, 53, 54, 55, 58, 59, 61, 63, 64, 69, 73, 74, 76, 77, 80]
            for video in a.keys():
                id = int(video[5:])
                if id not in valid_80:
                    continue
                video_folder = os.path.join(cholec80_frames,video)
                input = [(p["Frame_id"],p["Phase_gt"]) for p in a[video]]
                video_id = f"cholec80_{video}"
                for image,gt in input:
                    filename = f"{image}.jpg"
                    impath = os.path.join(video_folder,filename)
                    label = torch.zeros(self.class_num)
                    label[gt] = 1
                    data.append([impath,label,video_id])
            
            # For endoscapes
            def read_json(file_path):
                with open(file_path, 'r') as file:
                    data = json.load(file)
                return data
            
            endo_path = os.path.join(self.data_path,"endoscapes/test")
            b = read_json(os.path.join(endo_path,"annotation_ds_coco.json"))
            for p in b['images']:
                filename,gt,video = (p['file_name'],(p['ds']),p['video_id'])
                impath = os.path.join(endo_path,filename)
                gt = [round(value) for value in gt]
                label = torch.zeros(self.class_num)
                for i in range(len(gt)):
                    if gt[i] == 1:
                        label[i+7] = 1
                video_id = f"endoscapes_{video}"
                data.append([impath,label,video_id])
            
            # For cholect50
            test_split = [6,10,14,32,42,51,73,74,80,111]
            cholect50_frames = os.path.join(self.data_path,"cholect50/videos")
            cholect50_labels = os.path.join(self.data_path,"cholect50/labels")

            def get_label(labels):

                output = torch.zeros(self.class_num)
                for label in labels:
                    index = label[0]
                    if index == -1:
                        continue
                    output[index+10] = 1
                    phase = label[14]
                    output[phase] = 1
                    
                return output

            for i in test_split:
                video_folder = os.path.join(cholect50_frames,f"VID{i:02d}" if i < 100 else f"VID{i:03d}")
                label_file = os.path.join(cholect50_labels,f"VID{i:02d}.json" if i < 100 else f"VID{i:03d}.json")
                video_id = f"cholect50_{i}"

                with open(label_file, 'r') as file:
                    a = json.load(file)

                for frame_id,gts in a['annotations'].items():
                    filename = f"{int(frame_id):06d}.png"
                    impath = os.path.join(video_folder,filename)
                    label = get_label(gts)
                    data.append([impath,label,video_id])

            return data

    def name(self):
        return 'cholec'