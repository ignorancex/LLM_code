import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import torch
import nibabel as nib
import nibabel.processing
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable
import matplotlib
from math import exp
import ants
from nibabel.processing import resample_from_to


def extract_brain_with_mask_nib(brain_image_path, mask_image_path, output_image_path):
    """
    Apply a brain mask to a brain image to extract the brain using nibabel.

    Parameters:
    brain_image_path (str): The file path to the brain image.
    mask_image_path (str): The file path to the brain mask image.
    output_image_path (str): The file path where the extracted brain image will be saved.

    Returns:
    None: The function saves the extracted brain image to the specified output path.
    """
    
    # Load the brain image
    brain_img_nib = nib.load(brain_image_path)
    
    # Load the brain mask image
    mask_img_nib = nib.load(mask_image_path)
    
    # Get the image data as numpy arrays
    brain_data = brain_img_nib.get_fdata()
    mask_data = mask_img_nib.get_fdata()
    
    # Ensure the mask is boolean
    mask_data_bool = mask_data > 0
    
    # Apply the mask to the brain image data
    brain_data_masked = np.where(mask_data_bool, brain_data, 0)
    
    # Create a new NIfTI image with the masked data but the original header
    brain_extracted_nib = nib.Nifti1Image(brain_data_masked, brain_img_nib.affine, brain_img_nib.header)
    
    # Save the extracted brain image
    nib.save(brain_extracted_nib, output_image_path)

def resample_image_to_target_sparce(source_img_path, target_img_path, output_path, order):
    """
    Resamples the source image to the space of the target image using a specified interpolation order.

    Parameters:
    source_img_path (str): Path to the source image file.
    target_img_path (str): Path to the target image file.
    output_path (str): Path for saving the resampled source image.
    order (int): The order of the spline interpolation (default is 0, which is nearest-neighbor).
    """
    # Load source and target images
    source_img = nib.load(source_img_path)
    target_img = nib.load(target_img_path)

    # Get the shape and affine of the target image
    target_shape = target_img.shape
    target_affine = target_img.affine

    # Resample the source image to the target space
    resampled_img = resample_from_to(source_img, (target_shape, target_affine), order=order)

    # Save the resampled image
    nib.save(resampled_img, output_path)

def merge_nifti_labels(csf_path, gm_path, wm_path, output_path):
    """
    Merges three NIfTI files, assigning labels to each voxel based on the highest probability.
    Labels: 0 for background, 1 for CSF, 2 for GM, 3 for WM.

    Parameters:
        csf_path: Path to the CSF (Cerebrospinal Fluid) file
        gm_path: Path to the GM (Grey Matter) file
        wm_path: Path to the WM (White Matter) file
        output_path: Path for the output file
    """
    # Load NIfTI files
    csf_img = nib.load(csf_path)
    gm_img = nib.load(gm_path)
    wm_img = nib.load(wm_path)

    # Extract data
    csf_data = csf_img.get_fdata()
    gm_data = gm_img.get_fdata()
    wm_data = wm_img.get_fdata()

    # Combine data and add an extra dimension for background
    combined_data = np.stack((np.zeros(csf_data.shape), csf_data, gm_data, wm_data))

    # Compute the label with the highest probability
    labels = np.argmax(combined_data, axis=0)

    # Create a new NIfTI image
    new_img = nib.Nifti1Image(labels, csf_img.affine, csf_img.header)
    new_img.set_data_dtype(np.int16)

    # Save the new image
    nib.save(new_img, output_path)

    print(f"Merging complete, labeled image saved as '{output_path}'")

def resample_image_to_size(source_img_path, output_path, voxel_size, order):
    """
    Resamples the source image to the size using a specified interpolation order.

    Parameters:
    source_img_path (str): Path to the source image file.
    output_path (str): Path for saving the resampled source image.
    order (int): The order of the spline interpolation (default is 0, which is nearest-neighbor).
    """
    # Load source 
    source_img = nibabel.load(source_img_path)

    # Resample the source image to the size
    resampled_img = nibabel.processing.resample_to_output(source_img, voxel_size,order=order)
    
    resampled_raw_np=image_to_square_v2(resampled_img.get_fdata(),96)
    
    clipped_img = nib.Nifti1Image(resampled_raw_np, resampled_img.affine, nib.Nifti1Header())
    
    nib.save(clipped_img, output_path)
    
    return

def process_all_subjects_resample_to_96(base_path, voxel_size, output_base_path):
    for subject_folder in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject_folder)
        if os.path.isdir(subject_path) and subject_folder.startswith('sub-'):
            anat_folder = os.path.join(subject_path, 'anat')

            # Create the output folder if it does not exist
            if not os.path.exists(output_base_path):
                os.makedirs(output_base_path)

            # File paths and resampling for desc-preproc_T1w.nii.gz
            t1w_source_file = os.path.join(anat_folder, f"{subject_folder}_desc-preproc_T1w.nii.gz")
            t1w_output_file = os.path.join(output_base_path, f"{subject_folder}_desc-preproc_T1w_resample_to_96.nii.gz")
            if os.path.exists(t1w_source_file):
                resample_image_to_size(t1w_source_file, t1w_output_file, voxel_size, 1)

            # File paths and resampling for merged_labels_csf_gm_wm.nii.gz
            labels_source_file = os.path.join(anat_folder, f"{subject_folder}_merged_labels_csf_gm_wm.nii.gz")
            labels_output_file = os.path.join(output_base_path, f"{subject_folder}_merged_labels_csf_gm_wm_resample_to_96.nii.gz")
            if os.path.exists(labels_source_file):
                resample_image_to_size(labels_source_file, labels_output_file, voxel_size, 0)

            # File paths and resampling for aal_warped_in_subject_space.nii
            aal_source_file = os.path.join(anat_folder, f"{subject_folder}_aal_warped_in_subject_space.nii")
            aal_output_file = os.path.join(output_base_path, f"{subject_folder}_aal_warped_in_subject_space_resample_to_96.nii.gz")
            if os.path.exists(aal_source_file):
                resample_image_to_size(aal_source_file, aal_output_file, voxel_size, 0)

            # File paths and resampling for desc-brain_mask.nii.gz
            brain_mask_source_file = os.path.join(anat_folder, f"{subject_folder}_desc-brain_mask.nii.gz")
            brain_mask_output_file = os.path.join(output_base_path, f"{subject_folder}_desc-brain_mask_resample_to_96.nii.gz")
            if os.path.exists(brain_mask_source_file):
                resample_image_to_size(brain_mask_source_file, brain_mask_output_file, voxel_size, 0)

            print(f"Processed {subject_folder}")


def process_additional_files_to_orig_96(target_folder,voxel_size):
    additional_files = [
        ('./mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_brain.nii', 'mni_icbm152_t1_tal_nlin_asym_09c_brain_orig_96.nii', 1),
        ('./mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_csf_gm_wm_tal_nlin_asym_09c.nii', 'mni_icbm152_csf_gm_wm_tal_nlin_asym_09c_orig_96.nii', 0),
        ('./aal_for_SPM8/aal_for_SPM8/aal_resampled_in_mni152_space.nii', 'aal_resampled_in_mni152_space_orig_96.nii', 0)
    ]
    for input_file, output_name, order in additional_files:
        output_file = os.path.join(target_folder, output_name)
        resample_image_to_size(input_file, output_file, voxel_size, order)
    return
                
if __name__ == "__main__":
    extracted_brain = extract_brain_with_mask_nib("./mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii",
                                          "./mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii",
                                          "./mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_brain.nii")
    
    resample_image_to_target_sparce("./aal_for_SPM8/aal_for_SPM8/ROI_MNI_V4.nii", 
                         "./mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii",
                         "./aal_for_SPM8/aal_for_SPM8/aal_resampled_in_mni152_space.nii",
                               0)
    
    merge_nifti_labels('./mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_csf_tal_nlin_asym_09c.nii', 
                   './mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_gm_tal_nlin_asym_09c.nii', 
                   './mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_wm_tal_nlin_asym_09c.nii', 
                   './mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_csf_gm_wm_tal_nlin_asym_09c.nii')
    
    
    base_path="./fmriprep"
    voxel_size=[2.70, 2.70, 2.70]
    output_base_path="./ADHD_96_orig_space"

    process_all_subjects_resample_to_96(base_path, voxel_size, output_base_path)
    process_additional_files_to_orig_96(output_base_path,voxel_size)