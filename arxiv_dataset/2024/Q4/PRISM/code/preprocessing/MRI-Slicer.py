'''
````````` MRI Slicer `````````

Author: Sarang Galada
Description: This script slices MRI volumes and saves the middle slices of each as a .jpg image.
Input: NIfTI MRI volumes
Output: .jpg images of the middle axial slice of each MRI volume, in 256x256 resolution
Steps:
    1. Specify the input and output directories.
       Note: the input directory must have the following structure:
         > input_dir
            > site1
               > train
                  > MRI volume 1 (e.g., IXI060-Guys-709-T2.nii.gz)
                  > MRI volume 2
               > test
                  > MRI volume 1
                  > MRI volume 2
            > site2
               > train
                  > MRI volume 1
                  > MRI volume 2
               > test
                  > MRI volume 1
                  > MRI volume 2
            ...

    2. Specify the sites, modalities, and modes (train/test) to process.
    3. Specify if input MRI is skull stripped or not
    4. Run the script.
'''


import os
import numpy as np
from torchvision.transforms import Compose, Pad, CenterCrop, ToTensor, ToPILImage, Resize
import nibabel as nib
import matplotlib.pyplot as plt

# `````` TO DO ``````

## Specify the input and output directories
input_dir = "C:/Users/.../PRISM/pipeline/data folders/IXI-original-nifti-stripped/"
output_dir = "C:/Users/.../PRISM/pipeline/data folders/IXI-original-slices-stripped_T1/"

## Specify if the input MRI volumes are skull stripped or not
stripped = True

## Specify the sites, modalities, and modes to process
sites = ["Guys", "HH", "IOP"]
modalities = ["T2"]#, "T1", "PD"]
modes = ["train", "test"]

# `````` END TO DO ``````

default_transform = Compose([ToPILImage(), Pad(63), CenterCrop([256, 256])])
resize_transform = Compose([ToPILImage(), CenterCrop([253, 200]), Resize((256, 256))])

def background_removal(image):
    mask = image.ge(5e-2)
    image = image * mask
    return image

def nifti_slicer(in_path, out_path, modality):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for f in os.listdir(in_path):
        if (f.endswith(f"{modality}.nii.gz") and not stripped) or (f.endswith(f"{modality}_stripped.nii.gz") and stripped):
            subject_id = f[3:6]
            if not os.path.exists(out_path + f"/{subject_id}"):
                os.makedirs(out_path + f"/{subject_id}")

            if modality == "T1":
                vol = np.squeeze(nib.load(in_path + "/" + f).get_fdata().astype(np.float32))
                slice = vol[:, 150, :]
                # flip image horizontally
                slice = np.fliplr(slice)
                # pad zeros to the right edge of the image
                slice = np.pad(slice, ((0, 7), (5, 0)), mode='constant', constant_values=0)
                # max normalize
                slice = slice / np.max(slice)
                # resize image to 256x256
                slice = np.array(resize_transform(slice))
                # plt.imshow(slice, cmap='gray')
                # plt.axis('off')
                # plt.show()

            else:
                vol = np.squeeze(nib.load(in_path + "/" + f).get_fdata().astype(np.float32)) # load the nifti mri volume
                slice = np.rot90(vol[:, :, vol.shape[2] // 2]) # get the middle slice
                slice = slice / np.max(slice) # normalize the slice
                # plt.imshow(slice, cmap='gray')
                # plt.axis('off')
                # plt.show()
                
            slice = default_transform(slice) # apply default transform
            slice = ToTensor()(slice) # convert to tensor
            slice = background_removal(slice) # remove background
            if stripped:
                plt.imsave(out_path + f"/{subject_id}/IXI-{site}-{subject_id}-{modality}_stripped.png", slice.squeeze(), cmap='gray')
            else:
                plt.imsave(out_path + f"/{subject_id}/IXI-{site}-{subject_id}-{modality}.jpg", slice.squeeze(), cmap='gray')


# Process each site, mode, and modality
print(f"======== PRISM MRI Slicer ========\n")
print(f"Processing MRI volumes in the input directory: \n{input_dir}\n")
for site in sites:
    for mode in modes:
        for modality in modalities:
            in_path = input_dir + site + "/" + mode
            out_path = output_dir + site + "/" + mode
            nifti_slicer(in_path, out_path, modality)
    print(f"Finished processing site {site}")

print("Finished processing all sites.")
print(f"\nView the output images in the output directory: \n{output_dir}\n")