import cv2
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from torch.utils.data.dataset import Dataset


class FVUSMFramesDataset(Dataset):
    """ FVUSMDataset for FV-USM-processed
        root: Root dir for the FVUSM dataset.
        transform: Trsansforms for images.
        sample_per_class: How many samples in each class.
        mode: Train set or test set.
        inter_aug: Inter augmentation, 'LF' or 'TB'

    """

    def __init__(self, root, transforms=[], sample_per_class=6, mode='train', img_size=(256, 256), infer_dir='', rotate_infer_sample=False):
        self.transforms = transforms
        self.files = sorted(list(Path(root).rglob(f"*.*")))
        self.sample_per_class = sample_per_class
        self.class_num = len(self.files) // sample_per_class
        self.mode = mode
        self.rotate_infer_sample = rotate_infer_sample

        self.img_data = []
        for file_name in self.files:
            img = Image.open(file_name)
            img = img.convert('RGB')
            img = img.resize(img_size)
            self.img_data.append(img)
        
        # infer data
        self.infer_data = []
        self.infer_files = []
        if infer_dir:
            self.infer_files = sorted(list(Path(infer_dir).rglob("*.*")))
            for infer_file in self.infer_files:
                img = Image.open(infer_file)
                img = img.convert('RGB')
                img = img.resize(img_size)
                if self.rotate_infer_sample:
                    img = img.rotate(angle=90)
                self.infer_data.append(img)

    def __getitem__(self, index):
        out = {}
        if self.mode == 'train':
            sample_id_list = list(range(self.sample_per_class))
            source_idx = random.choice(sample_id_list)
            driving_idx = random.choice(sample_id_list)
            while True:
                if source_idx == driving_idx:
                    driving_idx = random.choice(sample_id_list)
                else:
                    break

            source = self.img_data[index * self.sample_per_class + source_idx]
            driving = self.img_data[index * self.sample_per_class + driving_idx]

            source = self.transforms(source)
            driving = self.transforms(driving)
            
            out['source'] = source
            out['driving'] = driving
        elif self.mode == 'val':
            if len(self.infer_data) != 0:
                # set source
                source = self.infer_data[index]
                out['source'] = (source.transpose(2, 0, 1) / 255.).astype(np.float32)
                # random select id from img_data
                index = random.choice(range(self.class_num))

            frames = []
            for i in range(self.sample_per_class):
                fid = index * self.sample_per_class + i
                image = self.img_data[fid]
                frames.append(image)

            frames = np.array(frames)
            frames = frames.transpose(0, 3, 1, 2) / 255.
            out['video'] = frames.astype(np.float32)
        elif self.mode == 'test':
            num_videos = len(self.img_data) // self.sample_per_class

            sources = []
            if len(self.infer_data) != 0:
                for source in self.infer_data:
                    sources.append(self.transforms(source))
                    break
            out['source'] = sources

            videos = []
            for video_start_idx in [self.sample_per_class * i for i in list(range(num_videos))]:
                frames = []
                for img_idx in range(video_start_idx, video_start_idx + self.sample_per_class):
                    image = self.img_data[img_idx]
                    image = self.transforms(image)
                    frames.append(image)

                frames = torch.stack(frames, dim=0)
                videos.append(frames)

            out['video'] = videos

        return out

    def __len__(self):
        if len(self.infer_files) != 0:
            # return len(self.infer_files)
            # TODO: Think about this!
            return 1
        else:
            return self.class_num
