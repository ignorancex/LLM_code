import os
import sys
import shutil
import math
import time
import h5py
from collections import Counter
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import nibabel as nib
import tkinter
import pandas as pd

def FindFolderNamePair(name):
    """
    Outputs patient number, moving and fixed image scanID as strings for further analysis.
    Possible folder name pairs are below with each string differing in length:
    name = '101117_CL_Dev_004_CL_Dev_008'
    name1 = '101117_CL_Dev_004_PS15_048'
    name2 = '101117_PS15_048_CL_Dev_004'
    name3 = '10097_PS15_048_PS17_017'

    Parameters:
        name (str): Input folder name.

    Returns:
        Tuple[str, str, str]: Patient number, moving image scanID, and fixed image scanID.
    """

    #idx contains a list of strings of a given folder name
    idx = [s for s in name.split("_")]

    if len(idx) == 6:
        if 'CL' in idx[1]:
            mov = f'{idx[1]}_{idx[2]}_{idx[3]}'
            fix = f'{idx[4]}_{idx[5]}'
            p = f'{idx[0]}'
            return(p, mov, fix)
        elif 'PS' in idx[1]:
            mov = f'{idx[1]}_{idx[2]}'
            fix = f'{idx[3]}_{idx[4]}_{idx[5]}'
            p = f'{idx[0]}'
            return(p, mov, fix)

    elif len(idx) == 7:
        mov = f'{idx[1]}_{idx[2]}_{idx[3]}'
        fix = f'{idx[4]}_{idx[5]}_{idx[6]}'
        p = f'{idx[0]}'
        return(p, mov, fix)

    elif len(idx) == 5:
        mov = f'{idx[1]}_{idx[2]}'
        fix = f'{idx[3]}_{idx[4]}'
        p = f'{idx[0]}'
        return(p, mov, fix)

    elif len(idx) == 4:
        mov = f'{idx[1]}'
        fix = f'{idx[2]}_{idx[3]}'
        p = f'{idx[0]}'
        return(p, mov, fix)


    else:
        print('Not a corresponding folder name')

def ThreeRegionsMapping():
    """
    Define a mapping of three brain regions (WM, GM, CSF) to corresponding anatomical structures.

    Returns:
        Dict[str, List[str]]: Mapping of brain regions to anatomical structures.
    """
    mapping = {
        'WM': ['cerebral white matter', 'brain-stem', 'cerebellum white matter', 'pallidum', 'ventral DC'],
        'GM': ['cerebral cortex', 'caudate', 'thalamus', 'putamen', 'hippocampus', 'amygdala', 'cerebellum cortex', 'accumbens area'],
        'CSF': ['CSF', 'lateral ventricle', '4th ventricle', '3rd ventricle', 'inferior lateral ventricle']
    }
    return mapping

def LabelsMappingSS():
    """
    Define a mapping of anatomical structure names to their corresponding label values in SynthSeg.

    Returns:
        Dict[str, List[int]]: Mapping of anatomical structure names to label values.
    """
    label_names = {
    'cerebral white matter': [2, 41],
    'cerebral cortex': [3, 42],
    'lateral ventricle': [4, 43],
    'inferior lateral ventricle': [5, 44],
    'cerebellum white matter': [7, 46],
    'cerebellum cortex': [8, 47],
    'thalamus': [10, 49],
    'caudate': [11, 50],
    'putamen': [12, 51],
    'pallidum': [13, 52],
    '3rd ventricle': [14],
    '4th ventricle': [15],
    'brain-stem': [16],
    'hippocampus': [17, 53],
    'amygdala': [18, 54],
    'CSF': [24],
    'accumbens area': [26, 58],
    'ventral DC': [28, 60]
    }  
    return label_names

def ExtractLabelsForRegions():
    """
    Extract values corresponding to anatomical regions from SynthSeg label mapping.

    Returns:
        Dict[str, List[int]]: Label values associated with White Matter (WM), Gray Matter (GM), and Cerebrospinal Fluid (CSF).
    """
    # Get the region mapping
    regions_mapping = ThreeRegionsMapping()
    
    # Get the label mapping
    labels_mapping = LabelsMappingSS()

    # Initialize dictionaries to store the values for each region
    values_by_region = {
        'WM': [],
        'GM': [],
        'CSF': []
    }

    # Iterate through each region and collect associated values
    for region, structures in regions_mapping.items():
        for structure in structures:
            if structure in labels_mapping:
                values_by_region[region].extend(labels_mapping[structure])

    return values_by_region

def BinarizeImage(image, seg):
    """
    Binarize a labeled image based on the specified anatomical segmentation region.

    Parameters:
        image (np.ndarray): Input labeled image.
        seg (str): Anatomical segmentation region ('WM', 'GM', 'CSF').

    Returns:
        np.ndarray: Binary mask representing the specified region (1 for True, 0 for False).
    """
    if seg == 'WM' or seg == 'GM' or seg == 'CSF':
        values_by_region = ExtractLabelsForRegions()

        labels_to_keep = values_by_region[seg]

        # Create a binary mask with 1s where the values match the specified labels
        binary_mask = np.isin(image, labels_to_keep)
        
        # Convert the boolean mask to integers (1 for True, 0 for False)
        binary_image = binary_mask.astype(int)
    
        return binary_image
    else:
        labels_names = LabelsMappingSS()
        labels_to_keep = list(labels_names.values())
        concatenated_list = sum(labels_to_keep, [])
        binary_mask = np.isin(image, concatenated_list)
        binary_image = binary_mask.astype(int)
    
        return binary_image

def DetJacobianPairRegionsDLANTs():
    """
    Calculate determinant of the Jacobian for all three main tissues using both DL and ANTs transformations.

    Reads command-line arguments for version, timestamp, and resolution. Has to be run with det-jacobian.sh.

    Returns:
        None
    """

    version = str(sys.argv[1]) #e.g. "RigidReg"
    time_stamp = str(sys.argv[2])#e.g. "20210527-222928_f0"
    res = str(sys.argv[3])#e.g. Resolution 1.5, 2,0 (in mm)

    fold = time_stamp[-2:] # equivalent to fo, f1 to f4

    segs = ['WM', 'GM', 'CSF']
    print(f'{version}_{fold}')
    for seg in segs:
        print(seg)
        pair_number = sorted(os.listdir(f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/"))
        remove_list=[]
        for i in pair_number:
            if i.endswith('.csv'):
                remove_list.append(i)
        for element in remove_list:
            if element in pair_number:
                pair_number.remove(element)

        folder_names = sorted(os.listdir(f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/"))
        # Create an empty DataFrame
        columns = ['pair_number', 'scanIDs', 'sum_abs_log_Jac_dl', 'non_zero_voxels_dl', 'sum_abs_log_Jac_ANTs', 'non_zero_voxels_ANTs', 'negative_dets_ANTs', 'negative_dets_dl' , 'percentage_negative_ANTs', 'percentage_negative_dl']
        df = pd.DataFrame(columns=columns, dtype=object)
        for pair_number in pair_number:
            idx = [int(s) for s in pair_number.split("_") if s.isdigit()]

            p, mov, fix = FindFolderNamePair(folder_names[idx[0]])

            fixed_path = f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/fix_seg.nii.gz"
            log_dl_path = f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/logJacobian.nii.gz"

            dl_ddf_path = f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/new_ddf.nii.gz"
            out_dl_path = f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/detJacobian.nii.gz"
            # os.system(f'CreateJacobianDeterminantImage 3 {dl_ddf_path} {out_dl_path} 0 1')

            detJa_dl = nib.load(out_dl_path).get_fdata()

            fixed_seg_img = nib.load(fixed_path)
            fix_hdr = fixed_seg_img.header.copy()
            fixed_seg = fixed_seg_img.get_fdata()
            fixed_seg_bin = BinarizeImage(fixed_seg, seg)
            
            dl_log_det_img = nib.load(log_dl_path)
            log_hdr = dl_log_det_img.header.copy()
            dl_log_det = dl_log_det_img.get_fdata()

            new_ddf_dl = fixed_seg_bin * dl_log_det
            new_det_Ja_dl = fixed_seg_bin * detJa_dl
            
            nbr_voxels_seg_dl = np.count_nonzero(fixed_seg_bin)
            nonzero_count_dl_det = np.count_nonzero(new_det_Ja_dl)
            
            sum_abs_dl = CalculateAbsSumLogJacobian(new_ddf_dl, type='sum_abs')
            comparison = np.where(np.array(new_det_Ja_dl) > 0, 0, np.array(new_det_Ja_dl)) #all negatives values are there and positives become 0
            negative_dets_dl = np.count_nonzero(comparison)
            percentage_negative_dl = 100 * (negative_dets_dl/nbr_voxels_seg_dl)

            if version == 'RigidReg':
                transfo = 'syn-for-rigid-1.5'
            elif version == 'RigidAffineReg':
                transfo = 'syn-1.5'
            else:
                transfo = 'only-syn-1.5'

            fixed_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_resc_new.nii.gz"
            
            log_ANTs_path = f'/media/andjela/SeagatePor/PairReg/{transfo}/images/{p}_{mov}_{fix}/logJacobian.nii.gz'

            ANTs_ddf_path = f"/media/andjela/SeagatePor/PairReg/{transfo}/images/{p}_{mov}_{fix}/ddf_ANTs.nii.gz"
            out_ANTs_path = f'/media/andjela/SeagatePor/PairReg/{transfo}/images/{p}_{mov}_{fix}/detJacobian.nii.gz'

            # os.system(f'CreateJacobianDeterminantImage 3 {ANTs_ddf_path} {out_ANTs_path} 0 1')

            detJa_ANTs = nib.load(out_ANTs_path).get_fdata()

            fixed_seg_img = nib.load(fixed_path)
            fix_hdr = fixed_seg_img.header.copy()
            fixed_seg = fixed_seg_img.get_fdata()
            fixed_seg_bin = BinarizeImage(fixed_seg, seg)
            
            ANTs_log_det = nib.load(log_ANTs_path).get_fdata()

            ANTs_log_det_img = nib.load(log_ANTs_path)
            log_hdr = ANTs_log_det_img.header.copy()
            ANTs_log_det = ANTs_log_det_img.get_fdata()

            new_ddf_ANTs = fixed_seg_bin * ANTs_log_det
            new_det_Ja_ANTs = fixed_seg_bin * detJa_ANTs

            nbr_voxels_seg_ANTs = np.count_nonzero(fixed_seg_bin)
            nonzero_count_ANTs_new = np.count_nonzero(detJa_ANTs)
            
            sum_abs_ANTs = CalculateAbsSumLogJacobian(new_ddf_ANTs, type= 'sum_abs')

            comparison = np.where(np.array(new_det_Ja_ANTs) > 0, 0, np.array(new_det_Ja_ANTs)) #all negatives values are there and positives become 0
            negative_dets_ANTs = np.count_nonzero(comparison)
            percentage_negative_ANTs = 100 * (negative_dets_ANTs/nbr_voxels_seg_ANTs)
            

            row = {'pair_number': pair_number, 'scanIDs': f'{mov}_{fix}', 'sum_abs_log_Jac_dl': sum_abs_dl, 'non_zero_voxels_dl': nbr_voxels_seg_dl, 'sum_abs_log_Jac_ANTs': sum_abs_ANTs, 'non_zero_voxels_ANTs': nbr_voxels_seg_ANTs, 'negative_dets_ANTs': negative_dets_ANTs, 'negative_dets_dl': negative_dets_dl, 'percentage_negative_ANTs':percentage_negative_ANTs, 'percentage_negative_dl':percentage_negative_dl }
            df = df.append(row, ignore_index=True) 
    
        
        df.to_csv(f"/media/andjela/SeagatePor/other_metrics/DetJa_{version}_{fold}_{res}/sum_abs_Det_Ja_{seg}")

def CalculateAbsSumLogJacobian(volume, type):
    """
    Calculate the absolute sum or sum of absolute of the log Jacobian values in the given volume.

    Parameters:
        volume (np.ndarray): Input 3D image volume containing log Jacobian values.
        type (str): Type of calculation ('sum_abs' for absolute sum, other for absolute sum of log JD).

    Returns:
        float: Absolute sum or sum of absolute of the log Jacobian values.
    """
    
    if type == 'sum_abs':
        abs_values = np.abs(volume)
        sum_absolute = np.sum(abs_values)

        return sum_absolute
    else:
        sum = np.sum(volume)
        abs_sum = np.abs(sum)

        return abs_sum


def DetJacobianDLANTsWhole():
    """
    Calculate the absolute sum of the log Jacobian values and other deformation related metrics for both DL and ANTs registration.
    Has to be called with det-jacobian.sh file.

    Returns:
        None
    """
    version = str(sys.argv[1]) #e.g. "RigidReg_run1"
    time_stamp = str(sys.argv[2])#e.g. "20210527-222928"
    res = str(sys.argv[3])#e.g. Resolution 1.5, 2.0 (in mm)

    
    fold = time_stamp[-2:] # equivalent to fo, f1 to f4

    pair_numbers = sorted(os.listdir(f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/"))
    remove_list=[] #remove .csv files in test folder
    for i in pair_numbers:
        if i.endswith('.csv'):
            remove_list.append(i)
    for element in remove_list:
        if element in pair_numbers:
            pair_numbers.remove(element)

    folder_names = sorted(os.listdir(f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/"))
    print(f'{version}_{res}')
    # Create an empty DataFrame
    columns = ['pair_number', 'scanIDs', 'sum_abs_log_Jac_dl', 'sum_abs_log_Jac_ANTs', 'nonzero_voxels_dl', 'nonzero_voxels_ANTs']
    df = pd.DataFrame(columns=columns, dtype=object)
    print(f'{version}_{fold}')
    for pair_number in pair_numbers:
        idx = [int(s) for s in pair_number.split("_") if s.isdigit()]

        p, mov, fix = FindFolderNamePair(folder_names[idx[0]])

        dl_ddf_path = f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/new_ddf.nii.gz"
        out_dl_path = f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/logJacobian.nii.gz"
        
        fixed_path = f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/fix_seg.nii.gz"
        fixed_seg = nib.load(fixed_path).get_fdata()
        fixed_seg_bin = BinarizeImage(fixed_seg, seg='all')
        # For creating determinant Jacobian volume for DL
        # os.system(f'CreateJacobianDeterminantImage 3 {dl_ddf_path} {out_dl_path} 1 1')
        dl_det = nib.load(out_dl_path).get_fdata()
        new_ddf_dl = fixed_seg_bin * dl_det
        nbr_voxels_seg_dl = np.count_nonzero(fixed_seg_bin)
        sum_abs_log_Jac_dl = CalculateAbsSumLogJacobian(new_ddf_dl, type='sum_abs') 
        
        if version == 'RigidReg':
            transfo = 'syn-for-rigid-1.5'
        elif version == 'RigidAffineReg':
            transfo = 'syn-1.5'
        else:
            transfo = 'only-syn-1.5'

        ANTs_ddf_path = f"/media/andjela/SeagatePor/PairReg/{transfo}/images/{p}_{mov}_{fix}/ddf_ANTs.nii.gz"
        out_ANTs_path = f'/media/andjela/SeagatePor/PairReg/{transfo}/images/{p}_{mov}_{fix}/logJacobian.nii.gz'

        fixed_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_resc_new.nii.gz"
        ANTs_det = nib.load(out_ANTs_path).get_fdata()
        fixed_seg = nib.load(fixed_path).get_fdata()
        fixed_seg_bin = BinarizeImage(fixed_seg, seg='all')
        new_ddf_ANTs = fixed_seg_bin * ANTs_det
        nbr_voxels_seg_ANTs = np.count_nonzero(fixed_seg_bin)
        # For creating determinant Jacobian volume for ANTs
        # os.system(f'CreateJacobianDeterminantImage 3 {ANTs_ddf_path} {out_ANTs_path} 1 1')
        
        sum_abs_log_Jac_ANTs = CalculateAbsSumLogJacobian(new_ddf_ANTs, type='sum_abs')

        row = {'pair_number': pair_number, 'scanIDs': f'{mov}_{fix}', 'sum_abs_log_Jac_dl': sum_abs_log_Jac_dl, 'sum_abs_log_Jac_ANTs': sum_abs_log_Jac_ANTs, 'nonzero_voxels_dl': nbr_voxels_seg_dl, 'nonzero_voxels_ANTs': nbr_voxels_seg_ANTs}
        df = df.append(row, ignore_index=True) 
    
    df.to_csv(f"/media/andjela/SeagatePor/other_metrics/DetJa_{version}_{fold}_{res}/sum_abs_log_Det_Ja_just_brain")

if __name__ == '__main__':

    # Select which one to run with det-jacobian.sh
    # DetJacobian within specific tissues
    DetJacobianPairRegionsDLANTs()

    # DetJacobian within the whole brain 
    # DetJacobianDLANTsWhole()


    