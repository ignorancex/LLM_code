import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import utils.utils_img as utils_img
import utils.utils as utils
import cv2
from PIL import Image

import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None, num_frames=16):
        self.video_dir = video_dir
        self.video_files = sorted(os.listdir(video_dir))  # Ensure consistent order
        self.transform = transform
        self.num_frames = num_frames  # Define number of frames per sample

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        frames_tensor = self.load_video_frames(video_path)
        return frames_tensor  # Shape: [16, 3, 512, 512]

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break  # Stop reading if the video ends
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)  # Apply transformation
            
            frames.append(frame)

        cap.release()

        # If the video has fewer frames than required, repeat or pad
        if len(frames) < self.num_frames:
            while len(frames) < self.num_frames:
                frames.append(frames[-1])  # Repeat the last frame

        # Select evenly spaced frames if too many
        elif len(frames) > self.num_frames:
            indices = torch.linspace(0, len(frames) - 1, self.num_frames).long()
            frames = [frames[i] for i in indices]

        # Convert list of frames into a tensor of shape [num_frames, 3, H, W]
        frames_tensor = torch.stack(frames)

        return frames_tensor
    

def get_dataloader_first_stage(args):
    print(f'>>> Loading data from {args.train_dir} and {args.val_dir}...')
    vqgan_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        utils_img.normalize_vqgan,
    ])
    train_loader = utils.get_dataloader(args.train_dir, vqgan_transform, args.batch_size, num_imgs=args.batch_size*args.train_steps, shuffle=True, num_workers=4, collate_fn=None)
    val_loader = utils.get_dataloader(args.val_dir, vqgan_transform, args.batch_size, num_imgs=args.batch_size*args.val_steps, shuffle=False, num_workers=4, collate_fn=None)
    
    print("Train Loader Length: ", len(train_loader))
    print("Val Loader Length: ", len(val_loader))
    return train_loader, val_loader

def get_dataloader_second_stage(args):
    transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.CenterCrop(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                utils_img.normalize_vqgan,
            ])

    train_dataset = VideoDataset(args.train_dir, transform)
    train_dataset = Subset(train_dataset, range(args.num_train_imgs))
    val_dataset = VideoDataset(args.val_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=None)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=None)

    print("Train Loader Length: ", len(train_loader))
    print("Val Loader Length: ", len(val_loader))
    return train_loader, val_loader

def get_dataloaders(args):
    if args.finetuning_stage == "first":
        train_loader, val_loader = get_dataloader_first_stage(args)
    elif args.finetuning_stage == "second":
        train_loader, val_loader = get_dataloader_second_stage(args)

    return train_loader, val_loader
    