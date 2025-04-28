import metat2.model.dmt.util.util as util
import os
import nibabel as nib
import numpy as np
from monai import transforms
import torch.utils.data as data


class ProstateDataset(data.Dataset):
    def __init__(self, image_dir, croot_modality='t2w', sroot_modality='dwi', max_images=6605):
        
        label_paths, image_paths, lesion_paths, instance_paths = self.get_paths(image_dir, croot_modality, sroot_modality)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        util.natural_sort(lesion_paths)

        for path1, path2, path3 in zip(label_paths, image_paths, lesion_paths):
            assert self.paths_match(path1, path2, path3), \
                "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths[:max_images]
        self.image_paths = image_paths[:max_images]
        self.lesion_paths = lesion_paths[:max_images]
        self.instance_paths = instance_paths[:max_images]

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, image_dir, croot_modality, sroot_modality):
        croot = os.path.join(image_dir, croot_modality)
        sroot = os.path.join(image_dir, sroot_modality)
        lesion = os.path.join(image_dir, 'lesion')

        c_image_paths = [os.path.join(croot, filename) for filename in os.listdir(croot)]
        s_image_paths = [os.path.join(sroot, filename) for filename in os.listdir(sroot)]
        lesion_paths = [os.path.join(lesion, filename) for filename in os.listdir(lesion)]

        instance_paths = []

        return c_image_paths, s_image_paths, lesion_paths, instance_paths

    def paths_match(self, path1, path2, path3):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        filename3_without_ext = os.path.splitext(os.path.basename(path3))[0]
        return filename1_without_ext == filename2_without_ext == filename3_without_ext

    def __getitem__(self, index):

        # Label (Content) Image
        label_path = self.label_paths[index]
        label = nib.load(label_path).get_fdata()
        label = np.expand_dims(label, axis=-1)

        # Real (Style) Image
        image_path = self.image_paths[index]
        image = nib.load(image_path).get_fdata()
        image = np.expand_dims(image, axis=-1)
        
        lesion_path = self.lesion_paths[index]
        lesion = nib.load(lesion_path).get_fdata()
        lesion = np.expand_dims(lesion, axis=-1)

        assert self.paths_match(label_path, image_path, lesion_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)

        image_transform = transforms.Compose([
            transforms.EnsureChannelFirst(channel_dim=-1),
            transforms.ScaleIntensity(minv=-1, maxv=1),
            transforms.ToTensor()
            ])

        lesion_transform = transforms.Compose([
            transforms.EnsureChannelFirst(channel_dim=-1),
            transforms.ToTensor()
            ])

        label_tensor = image_transform(label)
        image_tensor = image_transform(image)
        lesion_tensor = lesion_transform(lesion)

        input_dict = {'label': label_tensor, # t2w
                      'instance': 0,
                      'image': image_tensor, # dwi
                      'path': image_path,
                      'cpath': label_path,
                      'lesion': lesion_tensor
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
