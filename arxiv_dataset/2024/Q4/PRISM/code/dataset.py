### ====== PRISM MRI IMAGE DATASET ======
 
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Pad, CenterCrop, ToTensor, ToPILImage, Resize
import augmentations as aug
from PIL import Image
import glob
import os

default_transform = Compose([ToPILImage(), Pad(63), CenterCrop([256, 256])])

def get_fpaths(data_dir, mode, modality='T2', stripped=True):
        # directory structure: dataset_dir/{train, test}/{subject id}/{T2 image, PD image, T1 image, ...}
        fpaths = []
        if stripped:
            for subject_dir in glob(os.path.join(data_dir, mode, '*')):
                fpaths.extend(glob(os.path.join(subject_dir, f'*{modality}_stripped.jpg')))
            # for subject in os.listdir(os.path.join(data_dir, mode)):
            #     fpaths.append(data_dir+mode+'/'+subject)
        else:
            for subject_dir in glob(os.path.join(data_dir, mode, '*')):
                fpaths.extend(glob(os.path.join(subject_dir, f'*{modality}.jpg')))
                
        return fpaths


# Get MRI slice from jpg file path as 2d tensor
def get_mri_slice(fpath):
    if os.path.exists(fpath):
        slice = Image.open(fpath).convert('L')
        slice = ToTensor()(slice)
        # print(slice.shape)
    else:
        print("EMPTY IMAGE")
        slice = torch.ones([1, 256, 256])
        
    return slice

def get_mask(image):
    mask = torch.ones_like(image) * image.ge(5e-2)
    return mask.bool()


class PRISM_MRI_Dataset(Dataset):
    def __init__(self, data_dir, mode='train', modalities=['T2', 'PD'], stripped=True):
        self.data_dir = data_dir
        self.mode = mode
        self.modalities = modalities
        self.fpaths = get_fpaths(data_dir, mode, modalities[0], stripped)
        self.num_subjects = len(self.fpaths)

        self.dataset = []

        for i in range(self.num_subjects):
            image_dicts = []
            filename = self.fpaths[i].split('/')[-1] # eg. IXI-Guys-060-T2.jpg
            site_name = filename.split('-')[1] # eg. Guys
            subject_id = filename.split('-')[2] # eg. 060
            # site_name = "ADNI" # eg. ADNI
            # subject_id = filename[5:15]
            for modality in self.modalities:
                image_path = self.fpaths[i].replace('T2', modality)
                image = get_mri_slice(image_path)
                img = image.squeeze().numpy() * 255
                augs = [aug.gamma_correction(img, 0.5), aug.gamma_correction(img, 1.5), aug.apply_bias_fields(img, 3, 0.5, modality), aug.add_gaussian_noise(img, 0, 5)]
                augs = [ToTensor()(aug) for aug in augs]
                image_dict = {'image': image,
                            'aug': augs,
                            'mask': get_mask(image),
                            'site_name': site_name,
                            'subject_id': subject_id,
                            'modality': modality,
                            'exists': 0 if image[0, 0, 0] > 0.9999 else 1}
                image_dicts.append(image_dict)
            self.dataset.append(image_dicts)

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, i: int):
        return self.dataset[i]