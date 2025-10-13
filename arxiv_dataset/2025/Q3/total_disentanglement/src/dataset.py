import torch
import random
import cv2
import numpy as np
import os


class PretrainingDataset(torch.utils.data.Dataset):
    """Dataset class for pretraining with font-character disentanglement"""
    
    def __init__(self, img_dict, img_path_list, font_list, char_list, label, font, 
                 split, ext="png", transform=None, 
                 root_path='/workspace/dataset/google_fonts_imgs'):
        
        self.img_path_list = img_path_list
        self.transform = transform
        self.img_dict = img_dict
        self.font_list = font_list
        self.char_list = char_list
        self.root_path = root_path
        self.split = split
        
        self.label = label
        self.font = font
        self.ext = ext
        self.datanum = len(label)
    
    def __len__(self):
        return self.datanum
    
    def read_img(self, path):
        # Use the path directly from img_dict
        img = self.img_dict[path]
        
        # Extract font and character from path
        font_name = path.split("/")[-2]
        char_name = path.split("/")[-1].replace('.png', '')
        
        # Build paths for pair and ground truth using the same format as in img_dict
        base_dir = "/".join(path.split("/")[:-2])
        
        pair_font = random.choice(self.font_list)
        pair_char = random.choice(self.char_list)
        pair_path = f'{base_dir}/{pair_font}/{pair_char}.{self.ext}'
        gt_path = f'{base_dir}/{pair_font}/{char_name}.{self.ext}' 
        
        pair_img = self.img_dict[pair_path]
        gt_img = self.img_dict[gt_path]
        
        return img, pair_img, gt_img

    def __getitem__(self, idx):
        img, pair_img, gt_img = self.read_img(self.img_path_list[idx])
        label = self.label[idx]
        font = self.font[idx]
        
        if self.transform:
            img = self.transform(img)
            pair_img = self.transform(pair_img)
            gt_img = self.transform(gt_img)
            
        return img, label, font, pair_img, gt_img


class FinetuningDataset(torch.utils.data.Dataset):
    """Dataset class for finetuning"""
    
    def __init__(self, img, label, font, transform=None):
        self.transform = transform
        
        self.img = img
        self.label = label
        self.font = font
        self.datanum = len(label)
    
    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        img = self.img[idx]
        label = self.label[idx]
        font = self.font[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label, font


def load_pretraining_data(font, char_list, split='train', img_size=64, base_path=None):
    """Load data for pretraining
    
    Args:
        font: Font dataset name or path
        char_list: List of characters
        split: 'train' or 'valid'
        img_size: Image size
        base_path: Not used, kept for compatibility
    
    Returns:
        img_dict: Dictionary of images
        img_path_list: List of image paths
        font_list: List of font names
        char_list_full: Full list of characters
        ch_labels: Character labels
        font_labels: Font labels
    """
    ch_labels = []
    font_labels = []
    img_dict = {}
    font_list = []
    char_list_full = []
    img_path_list = []
    
    # Simple path construction
    font_path = f'./{font}/{split}/'
    
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Dataset path not found: {font_path}")
    
    font_dir_list = os.listdir(font_path)
    print(f"Loading from: {font_path}")
    
    for i in range(len(font_dir_list)):
        font_name = font_dir_list[i]
        if font_name == "failure.txt":
            continue
        font_list.append(font_name)
        
        for idx, ch in enumerate(char_list):
            char_list_full.append(str(ch))
            img_path = os.path.join(font_path, font_name, f'{str(ch)}.png')
            
            img = cv2.imread(img_path, 0)
            img_dict[img_path] = img
            
            img_path_list.append(img_path)
            ch_labels.append(idx)
            font_labels.append(int(i))
    
    char_list_full = list(set(char_list_full))
    
    return img_dict, img_path_list, font_list, char_list_full, ch_labels, font_labels


def load_finetuning_data(font, char_list, split='train', img_size=64, base_path=None):
    """Load data for finetuning
    
    Args:
        font: Font dataset name or path
        char_list: List of characters
        split: 'train' or 'valid'
        img_size: Image size
        base_path: Not used, kept for compatibility
    
    Returns:
        img_list: List of images
        ch_labels: Character labels
        font_labels: Font labels
        font_dir_list: List of font directories
    """
    img_list = []
    ch_labels = []
    font_labels = []
    
    # Simple path construction
    font_path = f'./{font}/{split}/'
    
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Dataset path not found: {font_path}")
    
    font_dir_list = os.listdir(font_path)
    print(f"Loading from: {font_path}")
    
    for i in range(len(font_dir_list)):
        font_name = font_dir_list[i]
        if font_name == "failure.txt":
            continue
            
        for idx, ch in enumerate(char_list):
            img_path = os.path.join(font_path, font_name, f'{str(ch)}.png')
            ch_labels.append(idx)
            font_labels.append(int(i))
            
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, dsize=(img_size, img_size))
            img = img / 255.0
            img_list.append(img)
    
    return img_list, ch_labels, font_labels, font_dir_list


def compute_font_features(model, font, char_list, split, img_size, device):
    """Compute average font features for each font
    
    Args:
        model: The trained model
        font: Font dataset name or path
        char_list: List of characters
        split: 'train' or 'valid'
        img_size: Image size
        device: Device to use
    
    Returns:
        font_features: Dictionary of average font features
        font_dir_list: List of font directories (excluding failure.txt)
    """
    font_path = f'./{font}/{split}/'
    font_dir_list = [d for d in os.listdir(font_path) if d != "failure.txt"]
    font_features = {}
    
    model.eval()
    with torch.no_grad():
        for font_idx, font_name in enumerate(font_dir_list):
            z_f_list = []
            for ch in char_list:
                img_path = os.path.join(font_path, font_name, f'{ch}.png')
                if os.path.exists(img_path):
                    img = cv2.imread(img_path, 0)
                    if img is not None:
                        img = cv2.resize(img, (img_size, img_size))
                        img = img / 255.0
                        img_torch = torch.from_numpy(img.astype(np.float32)).clone()
                        x = img_torch.reshape(1, 1, img_size, img_size).to(device, torch.float32)
                        
                        z_c, z_f, _ = model.encode(x)
                        z_f_list.append(z_f)
            
            if z_f_list:
                font_features[font_idx] = torch.mean(torch.stack(z_f_list), dim=0)
    
    return font_features, font_dir_list


def compute_char_features(model, font, char_list, split, img_size, device):
    """Compute average character features for each character
    
    Args:
        model: The trained model
        font: Font dataset name or path
        char_list: List of characters
        split: 'train' or 'valid'
        img_size: Image size
        device: Device to use
    
    Returns:
        char_features: Dictionary of average character features
    """
    font_path = f'./{font}/{split}/'
    font_dir_list = [d for d in os.listdir(font_path) if d != "failure.txt"]
    char_features = {}
    
    model.eval()
    with torch.no_grad():
        for char_idx, ch in enumerate(char_list):
            z_c_list = []
            for font_name in font_dir_list:
                img_path = os.path.join(font_path, font_name, f'{ch}.png')
                if os.path.exists(img_path):
                    img = cv2.imread(img_path, 0)
                    if img is not None:
                        img = cv2.resize(img, (img_size, img_size))
                        img = img / 255.0
                        img_torch = torch.from_numpy(img.astype(np.float32)).clone()
                        x = img_torch.reshape(1, 1, img_size, img_size).to(device, torch.float32)
                        
                        z_c, z_f, _ = model.encode(x)
                        z_c_list.append(z_c)
            
            if z_c_list:
                char_features[char_idx] = torch.mean(torch.stack(z_c_list), dim=0)
    
    return char_features