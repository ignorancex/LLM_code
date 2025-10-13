import os
import pickle
import json
import torch
from torchvision.transforms import Compose
from PIL import Image
import numpy as np
import random

from config import cfg
from log import logger

class Cholec_Val(torch.utils.data.Dataset):

    def __init__(self,sp,transform=None,f=False,class_num: int = -1):

        self.data_path = "cholec/data"
        self.transform = transform
        self.class_num = class_num
        self.is_sp = sp
        self.f = f
        data = self.read_data()
        self.data = data

    def read_data(self):

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
            if not self.f:
                if self.is_sp:
                    label = self.random_pick_one(label)
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
                if not self.f:
                    if self.is_sp:
                        label = self.random_pick_one(label)
                data.append([impath,label,video_id])

        return data
    
    def random_pick_one(self,tensor):

        ones_indices = torch.nonzero(tensor, as_tuple=False)
        
        if ones_indices.shape[0] > 0:
            random_idx = torch.randint((ones_indices.shape[0]), (1,))
            selected_index = ones_indices[random_idx]
            tensor.zero_()
            tensor[selected_index] = 1

        return tensor
    
    def __getitem__(self,index):
        impath, label, vid = self.data[index]
        image = Image.open(impath).convert("RGB")
        if self.f:
            if self.is_sp:
                label = self.random_pick_one(label)

        if self.transform:
            image = self.transform(image)
        
        if not self.f:
            if self.is_sp:
                assert torch.sum(label) < 2

        return image, label, vid
    
    def __len__(self):
        return len(self.data)

    def labels(self):
        with open('cholec/cholec_labels.txt', 'r')as f:
            text = f.read()
        return text.split('\n')

class Cholec_Train(torch.utils.data.Dataset):

    def __init__(self,transform=None,f=False,partial=False,class_num: int = -1):

        self.data_path = "cholec/data"
        self.transform = transform
        self.class_num = class_num
        self.f = f
        self.partial = partial
        self.data = self.read_data()

    def read_data(self):

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
            if not self.partial:
                if not self.f:
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
                if not self.partial:
                    if not self.f:
                        label = self.random_pick_one(label)
                data.append([impath,label,video_id])

        assert len(set([d[0] for d in data])) == len([d[1] for d in data]) == len(data)
        
        return data
    
    def random_pick_one(self,tensor):

        ones_indices = torch.nonzero(tensor, as_tuple=False)
        
        if ones_indices.shape[0] > 0:
            random_idx = torch.randint((ones_indices.shape[0]), (1,))
            selected_index = ones_indices[random_idx]
            tensor.zero_()
            tensor[selected_index] = 1

        return tensor
    
    def __getitem__(self,index):
        impath, label, vid = self.data[index]
        #print(label.sum())
        image = Image.open(impath).convert("RGB")
        if not self.partial:
            if self.f:
                label = self.random_pick_one(label)

        if self.transform:
            image = self.transform(image)

        if not self.partial:
            assert torch.sum(label) < 2
        
        return image, label, vid
    
    def __len__(self):
        return len(self.data)

    def labels(self):
        with open('cholec/cholec_labels.txt', 'r')as f:
            text = f.read()
        return text.split('\n')
    
class Cholec_Test(torch.utils.data.Dataset):

    def __init__(self,transform=None,class_num: int = -1):

        self.data_path = "cholec/data"
        self.transform = transform
        self.class_num = class_num
        self.data = self.read_data()

    def read_data(self):

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
    
    def __getitem__(self,index):
        impath, label, vid = self.data[index]
        image = Image.open(impath).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image , label, vid
    
    def __len__(self):
        return len(self.data)

    def labels(self):
        with open('cholec/cholec_labels.txt', 'r')as f:
            text = f.read()
        return text.split('\n')
    
class CholecDataset(torch.utils.data.Dataset):

    def __init__(self,label_list,transform=None,class_num: int = -1):

        self.transform = transform
        self.class_num = class_num
        self.data = label_list

    def __getitem__(self,index):
        impath, label, vid = self.data[index]
        image = Image.open(impath).convert("RGB")
        label = torch.Tensor(label)

        if self.transform:
            image = self.transform(image)

        return image , label, vid
    
    def __len__(self):
        return len(self.data)

    def labels(self):
        with open('cholec/cholec_labels.txt', 'r')as f:
            text = f.read()
        return text.split('\n')
    
class InitData(torch.utils.data.Dataset):

    def __init__(self,label_list,transform=None,class_num: int = -1):

        self.transform = transform
        self.class_num = class_num
        self.data = label_list

    def __getitem__(self,index):
        d = self.data[f"{index}"]
        impath = d['impath']
        label = d['label']
        image = Image.open(impath).convert("RGB")
        label = torch.Tensor(label)

        if self.transform:
            image = self.transform(image)

        return image , label
    
    def __len__(self):
        return len(self.data)

    def labels(self):
        with open('cholec/cholec_labels.txt', 'r')as f:
            text = f.read()
        return text.split('\n')

def build_cholec_dataset(train_preprocess: Compose,
                       val_preprocess: Compose,
                       pin_memory=True):
    
    if cfg.use_lfile:

        logger.info(f'Using label file....{cfg.label_file}')
        label_file = cfg.label_file
        a = open(label_file,'r')
        f = json.load(a)
        a.close()

        datasets = {}
        for key in f.keys():
            if key not in datasets:
                datasets[key] = []
            
            for ki in f.get(key):
                datasets[key].extend([[k,v.get('label'), v.get('videoID')] for k,v in f[key][ki].items() if os.path.exists(k)])

            
        if cfg.val_sp:
            logger.info("Loading single positive validation")
            val_sp_dataset = CholecDataset(datasets['val_sp'],val_preprocess,class_num=cfg.num_classes)
        logger.info("Loading partial positive validation")
        val_dataset = CholecDataset(datasets['val_full'],val_preprocess,class_num=cfg.num_classes)
        train_dataset = CholecDataset(datasets['train'],train_preprocess,class_num=cfg.num_classes)
        test_dataset = CholecDataset(datasets['test'],val_preprocess,class_num=cfg.num_classes)
        
    else:
        if cfg.getitem:
            logger.info('Using getitem....')
        else:
            logger.info('Using init labels....')

        val_dataset = Cholec_Val(False,val_preprocess, cfg.getitem, class_num=cfg.num_classes)
        val_sp_dataset = Cholec_Val(True,val_preprocess, cfg.getitem, class_num=cfg.num_classes)
        train_dataset = Cholec_Train(train_preprocess, cfg.getitem, cfg.partial, class_num=cfg.num_classes)
        test_dataset = Cholec_Test(val_preprocess,class_num=cfg.num_classes)


    if cfg.perform_init:

        init_train_file = cfg.init_train_file
        b = open(init_train_file,'r')
        init_train_data = json.load(b)
        b.close()

        init_train_dataset = InitData(init_train_data,train_preprocess,class_num=cfg.num_classes)

        init_val_file = cfg.init_val_file
        c = open(init_val_file,'r')
        init_val_data = json.load(c)
        c.close()

        init_val_dataset = InitData(init_val_data,val_preprocess,class_num=cfg.num_classes)

    if cfg.perform_init:
        logger.info(f"Length of init train data : {len(init_train_dataset)}")
        logger.info(f"Length of init val data : {len(init_val_dataset)}")
    logger.info(f"Length of train : {len(train_dataset)}")
    logger.info(f"Length of val : {len(val_dataset)}")
    if cfg.val_sp:
        logger.info(f"Length of val sp : {len(val_sp_dataset)}")
    logger.info(f"Length of test : {len(test_dataset)}")

    total_ones = 0
    for _, label, _ in train_dataset.data:
        total_ones += torch.sum(label).item()  # Sum the 1s in the label tensor
    
    print(f"Total ones: {total_ones}")

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(  # type: ignore
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers = True)

    val_loader = torch.utils.data.DataLoader(  # type: ignore
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers = True)
    
    if cfg.val_sp:
        val_sp_loader = torch.utils.data.DataLoader(  # type: ignore
            val_sp_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.workers,
            pin_memory=False,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers = True)
        
    test_loader = torch.utils.data.DataLoader(  # type: ignore
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers = True)
    
    if cfg.perform_init:
        init_train_loader = torch.utils.data.DataLoader(  # type: ignore
            init_train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.workers,
            pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers = True)
        
        init_val_loader = torch.utils.data.DataLoader(  # type: ignore
            init_val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.workers,
            pin_memory=False,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers = True)

    logger.info("Build dataset done.")
    if cfg.perform_init:
        if cfg.val_sp:
            return [train_loader, val_loader, val_sp_loader, test_loader, init_train_loader, init_val_loader]
        else:
            return [train_loader, val_loader, test_loader, init_train_loader, init_val_loader]
    else:
        if cfg.val_sp:
            return [train_loader, val_loader, val_sp_loader, test_loader]
        else:
            return [train_loader, val_loader, test_loader]