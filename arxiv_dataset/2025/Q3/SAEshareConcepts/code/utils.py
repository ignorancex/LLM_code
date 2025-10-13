import datasets
import torch
from torch.utils.data import DataLoader, Dataset

import os
import shutil
from glob import glob
from tqdm import tqdm
from PIL import Image

class Flowers102ImagesDataset(Dataset):
    def __init__(self,  transform):
        self.dataset = datasets.load_dataset("efekankavalci/flowers102-captions")['train']
        self.transform = transform
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        transformed_image = self.transform(image)
        return transformed_image
    
class Flowers102TextDataset(Dataset):
    def __init__(self,):
        self.dataset = datasets.load_dataset("efekankavalci/flowers102-captions")['train']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = str(self.dataset[idx]["text"])
        return text


def get_dataloader_flowersimage(transform, batch_size=128, shuffle=True, num_workers=0):
    dataset = Flowers102ImagesDataset(transform)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=False,
        shuffle=shuffle,
        pin_memory=False,
    )

def get_dataloader_flowerstext(batch_size=128, shuffle=True, num_workers=0):
    dataset = Flowers102TextDataset()
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=False,
        shuffle=shuffle,
        pin_memory=False,
    )


class CocoImagesDataset(Dataset):
    def __init__(self,  transform):
        self.dataset = datasets.load_dataset("wangherr/coco2017_train_image_caption")['train']
        self.transform = transform
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        if self.transform is None: return image
        try:
            transformed_image = self.transform(image)
            return transformed_image
        except:
            return self.transform(Image.new('RGB', (400,400)))
        
class CocoTextDataset(Dataset):
    def __init__(self,):
        self.dataset = datasets.load_dataset("wangherr/coco2017_train_image_caption")['train']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = str(self.dataset[idx]["text"])
        return text


def get_dataloader_cocoimage(transform, batch_size=128, shuffle=True, num_workers=0):
    dataset = CocoImagesDataset(transform)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=False,
        shuffle=shuffle,
        pin_memory=False,
    )

def get_dataloader_cocotext(batch_size=128, shuffle=True, num_workers=0):
    dataset = CocoTextDataset()
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=False,
        shuffle=shuffle,
        pin_memory=False,
    )


def concatenate_run(run):
    """
    Concatenate the recorded SAE features, to keep only one artifact file per layer    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = [
        f'sae_features/{run}/layer{str(i).zfill(2)}' for i in range(24)
    ]
    layers.append(f'sae_features/{run}/layer-1')
    layers = layers[::-1]
    for layer in tqdm(layers, desc='Cleaning'):
        try:
            files = glob(f'{layer}/*.pt')
            
            nb = [int(f.split('/')[-1].split('.')[0]) for f in files]
            files = [f for _,f in sorted(zip(nb, files), key=lambda x: x[0])]
            if len(files) == 0: continue
            layer_name = layer.split('/')[-1]
            if len(layer_name) < 2: layer_name = layer.split('/')[-1]
            layer_name = str(layer_name) + '.pt'
            if os.path.exists(f'sae_features/{run}/{layer_name}'): continue
            arr = torch.vstack([torch.load(f, map_location=device).to(torch.float16) for f in files]).to_dense()
            torch.save(arr, f'sae_features/{run}/{layer_name}')
            del arr
        except: pass

    subfolders = glob(f"sae_features/{run}/layer*/")
    for f in subfolders: shutil.rmtree(f)