import os
import random
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import resize
from tqdm import tqdm

# Ensure np.int is correctly mapped (to avoid deprecated warnings)
np.int = int  # Alias np.int to int

class CoCoDataset(data.Dataset):
    def __init__(
        self,
        image_dir,
        anno_path,
        num_classes,  # Dynamically set class number
        input_transform=None,
        labels_path=None,
        used_category=-1,
        train_classes=None,
        seathru_transform=None,
        resize=(384, 384)  # Define resize dimensions
    ):
        self.coco = dset.CocoDetection(root=image_dir, annFile=anno_path)
        self.num_classes = num_classes  # Store dynamic class number
        self.category_map = {str(i): i+1 for i in range(self.num_classes)}  # Create dynamic category map
        self.input_transform = input_transform
        self.labels_path = labels_path
        self.used_category = used_category
        self.train_classes = train_classes
        self.seathru_transform = seathru_transform
        self.resize = resize  # Resize dimensions

        self.load_labels()
        if self.train_classes is not None:
            self.filter_samples(self.train_classes)

        self.resize_transform = transforms.Resize(self.resize)

    def load_labels(self):
        """Load or compute multi-hot encoded labels for the dataset."""
        if os.path.exists(self.labels_path):
            self.labels = np.load(self.labels_path).astype(np.float32)
        else:
            print(f"No preprocessed label file found in {self.labels_path}. Computing labels...")
            self.labels = []
            for i in tqdm(range(len(self.coco))):
                _, annotations = self.coco[i]
                categories = self.getCategoryList(annotations)
                label = self.getLabelVector(categories)
                self.labels.append(label)
            self.save_datalabels(self.labels_path)

    def filter_samples(self, train_classes):
        """Filter samples to include only specified training classes."""
        filtered_indices = [i for i, label in enumerate(self.labels) if any(label[class_idx] == 1 for class_idx in train_classes)]
        self.filtered_coco = [self.coco[i] for i in filtered_indices]
        self.labels = [self.labels[i] for i in filtered_indices]

    def __getitem__(self, index):
        item = self.filtered_coco[index] if hasattr(self, 'filtered_coco') else self.coco[index]
        image, _ = item
        img_info = self.coco.coco.imgs[self.coco.ids[index]]
        img_name = img_info['file_name']

        # Convert to PIL Image before resizing
        image = Image.fromarray(np.array(image))

        # Resize the image
        image = resize(image, self.resize)

        # Apply SeaThru transform if available
        if self.seathru_transform:
            image = self.seathru_transform(img_name, np.array(image))

        # Ensure correct format (C, H, W)
        if image.ndim == 3 and image.shape[0] != 3:
            image = image.permute(2, 0, 1)

        return image, torch.tensor(self.labels[index])


    def getCategoryList(self, item):
        """Extract category IDs from annotations."""
        return list(set(t["category_id"] for t in item))

    def getLabelVector(self, categories):
        """Create a multi-hot encoded vector for the given categories."""
        label = [0] * self.num_classes  # Adjusted for dynamic category size
        for c in categories:
            index = self.category_map.get(str(c))
            if index is not None and index < len(label):
                label[index] = 1.0
        return np.array(label, dtype=np.int32)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.filtered_coco) if hasattr(self, 'filtered_coco') else len(self.coco)

    def save_datalabels(self, outpath):
        """Save multi-hot encoded labels to a file."""
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        np.save(outpath, np.array(self.labels))