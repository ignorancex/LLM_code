# from pytorch_lightning import LightningDataModule
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from src.datamodules.components.video_dataset import VideoDataset
# import torch


# class VideoDataModule(LightningDataModule):
#     def __init__(self, train_file, test_file, num_segments=16, batch_size=16, num_workers=4):
#         super().__init__()
#         self.train_file = train_file
#         self.test_file = test_file
#         self.num_segments = num_segments
#         self.batch_size = batch_size
#         self.num_workers = num_workers
        
#         self.transform_train = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.1),
#             transforms.ToTensor(),
#             transforms.RandomErasing(p=0.7, scale=(0.04, 0.5)),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
        
#         self.transform_val = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])

#     def setup(self, stage=None):
#         self.train_dataset = VideoDataset(
#             list_file=self.train_file,
#             num_segments=self.num_segments,
#             transform=self.transform_train,
#             random_shift=True,
#             data_folder=None,
#         )
#         self.val_dataset = VideoDataset(
#             list_file=self.test_file,
#             num_segments=self.num_segments,
#             transform=self.transform_val,
#             random_shift=False,
#             data_folder=None,
#         )

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             pin_memory=True,
#             collate_fn=self.custom_collate_fn
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=True,
#             collate_fn=self.custom_collate_fn
#         )

#     def custom_collate_fn(self, batch):
#         if len(batch[0]) == 3:  # Training data
#             global_inputs = [item[0] for item in batch]
#             alt_global_inputs = [item[1] for item in batch]
#             labels = [item[2] for item in batch]
#             return torch.stack(global_inputs), torch.stack(alt_global_inputs), torch.tensor(labels)
#         else:  # Validation data
#             inputs = [item[0] for item in batch]
#             labels = [item[1] for item in batch]
#             return torch.stack(inputs), torch.tensor(labels)

#     def test_dataloader(self):
#         return self.val_dataloader()

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from src.datamodules.components.video_dataset import VideoDataset
import torch

class VideoDataModule(LightningDataModule):
    def __init__(self, train_file, test_file, num_segments=16, batch_size=16, num_workers=4):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.num_segments = num_segments
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.7, scale=(0.04, 0.5)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        self.train_dataset = VideoDataset(
            list_file=self.train_file,
            num_segments=self.num_segments,
            transform=self.transform_train,
            random_shift=True,
            data_folder=None,
        )
        self.val_dataset = VideoDataset(
            list_file=self.test_file,
            num_segments=self.num_segments,
            transform=self.transform_val,
            random_shift=False,
            data_folder=None,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.custom_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.custom_collate_fn
        )

    def custom_collate_fn(self, batch):
        if len(batch[0]) == 4:  # Training data
            global_inputs = [item[0] for item in batch]
            alt_global_inputs = [item[1] for item in batch]
            labels = [item[2] for item in batch]
            indices = [item[3] for item in batch]
            return torch.stack(global_inputs), torch.stack(alt_global_inputs), torch.tensor(labels), torch.tensor(indices)
        else:  # Validation data
            inputs = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            indices = [item[2] for item in batch]
            return torch.stack(inputs), torch.tensor(labels), torch.tensor(indices)

    def test_dataloader(self):
        return self.val_dataloader()