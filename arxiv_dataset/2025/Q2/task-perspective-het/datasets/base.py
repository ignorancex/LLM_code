import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from .utils import return_preprocess

class BaseMultiTaskDataset(Dataset):

    def __init__(self,
                 data_fps,
                 label_fps,
                 src_task,
                 dest_tasks):
        
        self.src_task = src_task
        self.dest_tasks = dest_tasks
        self.data = data_fps
        self.labels = {}
        for task in self.dest_tasks:
            self.labels[task] = label_fps[task]
        self.datalen = len(self.data)
        self.src_transform = return_preprocess(self.src_task)
        self.dest_transforms = {task: return_preprocess(task) for task in self.dest_tasks}
    
    def __len__(self):
        return self.datalen
    
    def __getitem__(self, idx):

        data = Image.open(self.data[idx])
        data = self.src_transform(data)
        label = {}

        for task in self.dest_tasks:

            if task in ["class_scene"]:
                label[task] = self.dest_transforms[task](self.labels[task][idx]).long()
            
            else:
                label[task] = Image.open(self.labels[task][idx])
                label[task] = self.dest_transforms[task](label[task])
                if task =="segment_semantic":
                    label[task] = label[task].long().unsqueeze(0)
        return data, label