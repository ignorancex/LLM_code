from torch.utils.data import Dataset
import os
from .function import *
from PIL import Image
from torchvision import transforms



class Vein(Dataset):
    def __init__(self, args, mode = 'train', transform = None):
        self.root_dir = args.data_path
        self.videos = []
        self.transform = transform
        self.img_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])
        frames_root = os.path.join(self.root_dir, mode, 'images')
        for video_name in sorted(os.listdir(frames_root), key=sort_key):
            video_path = os.path.join(frames_root, video_name)
            img_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path)], key=sort_key)
            self.videos.append(img_files)

    
    def __len__(self):
        return sum([len(img_files) for img_files in self.videos])

    
    def __getitem__(self, index):
        for video_id, img_files in enumerate(self.videos):
            if index < len(img_files):
                img_path = img_files[index]
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(img_path.replace('images', 'masks').replace('jpg', 'png')).convert('L').point(lambda x: 1 if x > 0 else 0, mode='1')
                break
            index -= len(img_files)
        
        if self.transform:
            img, mask = self.transform(img, mask)
        img, mask = self.img_transform(img), self.gt_transform(mask)
        
        return {
            'images': img,
            'masks': mask,
            'video_id': video_id
        }



class SUN_SEG(Dataset):
    def __init__(self, args, mode = 'train', transform = None):
        self.root_dir = args.data_path
        self.videos = []
        self.transform = transform
        self.img_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])
        if mode == 'train':
            mode = 'TrainDataset'
        else:
            mode = 'TestEasyDataset/Unseen'
        frames_root = os.path.join(self.root_dir, mode, 'Frame')
        for video_name in sorted(os.listdir(frames_root), key=sort_key):
            video_path = os.path.join(frames_root, video_name)
            img_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path)], key=sort_key)
            self.videos.append(img_files)

    
    def __len__(self):
        return sum([len(img_files) for img_files in self.videos])

    
    def __getitem__(self, index):
        for video_id, img_files in enumerate(self.videos):
            if index < len(img_files):
                img_path = img_files[index]
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(img_path.replace('Frame', 'GT').replace('jpg', 'png')).convert('L').point(lambda x: 1 if x > 0 else 0, mode='1')
                break
            index -= len(img_files)
        
        if self.transform:
            img, mask = self.transform(img, mask)
        img, mask = self.img_transform(img), self.gt_transform(mask)
        
        return {
            'images': img,
            'masks': mask,
            'video_id': video_id
        }