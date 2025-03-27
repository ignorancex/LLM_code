import torch

from pathlib import Path

class SlideDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        slide_data: list,
        dataset_name: str,
        base_folder: str,
    ):
        self.slide_data = slide_data
        self.dataset_name = dataset_name
        self.base_folder = base_folder

    def __getitem__(self, idx: int):
        subset, class_, slide = self.slide_data[idx]
        if self.dataset_name == "camelyon16":
            slide_dir = f"{self.base_folder}/{subset}/{class_}/{slide}/imgs"
            patches = [p for p in Path(slide_dir).glob('*.jpg')]
        elif self.dataset_name == "tcga-nsclc":
            slide_dir = f"{self.base_folder}/{subset}/{class_}/{slide}"
            inner_patches = [p for p in Path(slide_dir).glob('*/*.jpg')]
            outer_patches = [p for p in Path(slide_dir).glob('*.jpg')]
            patches = inner_patches + outer_patches

        return idx, patches

    def __len__(self):
        return len(self.slide_data)