import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from otdd import DatasetDistance

# Define dataset directories
midjourney_dir = '/scratch/qilong3/transferattack/data/mjimages'
dalle2_dir = '/scratch/qilong3/datasets/DALLE2/train'
diffusiondb_dir = '/scratch/qilong3/datasets/DiffusionDB/train'  # Example path for DiffusionDB

# Define transformations for images (resize to the same size, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Custom Dataset class for unlabeled images
class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith(('jpg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Load the datasets using the custom loader
midjourney_dataset = UnlabeledImageDataset(midjourney_dir, transform=transform)
dalle2_dataset = UnlabeledImageDataset(dalle2_dir, transform=transform)
diffusiondb_dataset = UnlabeledImageDataset(diffusiondb_dir, transform=transform)

# Create a DataLoader for each dataset
batch_size = 32
midjourney_loader = DataLoader(midjourney_dataset, batch_size=batch_size, shuffle=True)
dalle2_loader = DataLoader(dalle2_dataset, batch_size=batch_size, shuffle=True)
diffusiondb_loader = DataLoader(diffusiondb_dataset, batch_size=batch_size, shuffle=True)

# Define a function to compute OTDD between two datasets
def compute_otdd(loader1, loader2, device='cuda'):
    distance = DatasetDistance(
        loader1,
        loader2,
        inner_ot_method='exact',
        debiased_loss=True,
        p=2,
        entreg=1e-1,
        device=device
    )
    dist = distance.distance()
    return dist

# Compute OTDD between Midjourney and DALLE-2
midjourney_dalle2_otdd = compute_otdd(midjourney_loader, dalle2_loader)
print(f"OTDD between Midjourney and DALLE-2 datasets: {midjourney_dalle2_otdd}")

# Compute OTDD between Midjourney and DiffusionDB
midjourney_diffusiondb_otdd = compute_otdd(midjourney_loader, diffusiondb_loader)
print(f"OTDD between Midjourney and DiffusionDB datasets: {midjourney_diffusiondb_otdd}")

# Compute OTDD between DALLE-2 and DiffusionDB
dalle2_diffusiondb_otdd = compute_otdd(dalle2_loader, diffusiondb_loader)
print(f"OTDD between DALLE-2 and DiffusionDB datasets: {dalle2_diffusiondb_otdd}")