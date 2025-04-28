import os
import sys
import shutil
import math
import time
import numpy as np
import nibabel as nib
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

def dice(array1, array2, labels=None, include_zero=False):
    """
    Function from Voxelmorph to compute Dice score for many labels from:
    https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/py/utils.py 
    Computes the dice overlap between two arrays for a given set of integer labels.

    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0)) 

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem

def DiceScoresANTsSynthSeg():
    """
    Function to calculate Dice Scores for a given ANTs initialization version (NoReg, RigidReg or RigidAffineReg) by first warping synthseg 
    segmentations using obtained ANTs composite transform files.
    To be called with warp-and-dice-ANTs-synthseg.sh file. 
    """

    version = str(sys.argv[1]) #e.g. "RigidReg"
    time_stamp = str(sys.argv[2])#e.g. "20210527-222928_f0"
    res = str(sys.argv[3])#e.g. Resolution 1.5, 2.0 (in mm)

    
    fold = time_stamp[-2:] # equivalent to fo, f1 to f4

    dice_means = []
    reg_times = []

    pair_numbers = sorted(os.listdir(f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/"))
    #Remove .csv files in test folder
    remove_list=[] 
    for i in pair_numbers:
        if i.endswith('.csv'):
            remove_list.append(i)
    for element in remove_list:
        if element in pair_numbers:
            pair_numbers.remove(element)

    folder_names = sorted(os.listdir(f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/"))

    # Create an empty DataFrame
    columns = ['pair_number', 'scanIDs', 'label', 'Dice Score']
    df = pd.DataFrame(columns=columns, dtype=object)

    if not os.path.exists(f"/media/andjela/SeagatePor/warped_segs/SynthSeg/ANTs/{version}_{res}/"):
        os.mkdir(f"/media/andjela/SeagatePor/warped_segs/SynthSeg/ANTs/{version}_{res}/")

    for pair_number in pair_numbers:
        idx = [int(s) for s in pair_number.split("_") if s.isdigit()]

        p, mov, fix = FindFolderNamePair(folder_names[idx[0]])

        # Take fix and moving images that were already rescaled in DiceScoresDLSynthSeg()
        fix_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_resc_new.nii.gz"
        fixed_seg = nib.load(fix_path).get_fdata()

        if version == 'RigidReg':
            transfo = 'syn-for-rigid-1.5'
        elif version == 'RigidAffineReg':
            transfo = 'syn-1.5'
        else:
            transfo = 'only-syn-1.5'

        # Warp images with obtained ANTs composite file
        mov_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{mov}_resc_new.nii.gz"
        transform_path = f"/media/andjela/SeagatePor/PairReg/{transfo}/images/{p}_{mov}_{fix}/mov2fix_Composite.h5"

        
        out_path = f"/media/andjela/SeagatePor/warped_segs/SynthSeg/ANTs/{version}_{res}/warped_{pair_number}.nii.gz"

        # Predict warp and time for ANTs
        start = time.time()

        # Adjust verbose to 1 if output for ANTs is needed
        os.system(f"antsApplyTransforms --default-value 0 --float 0 " \
            f"--input {mov_path} " \
            f"--interpolation GenericLabel " \
            f"--input-image-type 0 " \
            f"--output {out_path} " \
            f"--reference-image {fix_path} " \
            f"--verbose 0 " \
            f"--transform {transform_path}"
            )
        
        reg_time = time.time() - start
        reg_times.append(reg_time)

        warped_seg = nib.load(out_path).get_fdata()

        # Following SynthSeg documentation where numbers for segmented regions do not follow normal numbering
        labels = np.array([0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60])
        # 0 only represents the backgroung, hence it is excluded
        dice_scores = dice(warped_seg, fixed_seg, labels=labels, include_zero=False)
        
        dice_means.append(np.mean(dice_scores))

        labels = np.delete(labels, np.argwhere(labels == 0))
        for label in labels:
            row = {'pair_number': pair_number, 'scanIDs': f'{mov}_{fix}', 'label': label, 'Dice Score': dice_scores[np.where(labels == label)[0]][0]}
            df = df.append(row, ignore_index=True) 

        print('Pair nb %s   scanIDs %s    Reg Time: %.4f    Dice: %.4f +/- %.4f' % (pair_number, f'{mov}_{fix}', reg_time,
                                                                    np.mean(dice_scores),
                                                                    np.std(dice_scores)))

    if not os.path.exists(f"/media/andjela/SeagatePor/dice_scores/SynthSeg/ANTs/{version}_{res}_{fold}/"):
        os.mkdir(f"/media/andjela/SeagatePor/dice_scores/SynthSeg/ANTs/{version}_{res}_{fold}/")

    df.to_csv(f"/media/andjela/SeagatePor/dice_scores/SynthSeg/ANTs/{version}_{res}_{fold}/dice_scores_{version}")
    print()
    print(f'{version}_{fold}')
    print('Avg Reg Time: %.4f +/- %.4f  (skipping first prediction)' % (np.mean(reg_times),
                                                                        np.std(reg_times)))
    print('Avg Dice: %.4f +/- %.4f' % (np.mean(dice_means), np.std(dice_means)))

def FlirtDownsampling(img_path, ref_path, out_seg_path):
    """
    Downsample a segmentation image using FSL FLIRT tool with nearest neighbor interpolation.

    Parameters:
        - img_path (str): Path to the input segmentation image.
        - ref_path (str): Path to the reference image for downsampling.
        - out_seg_path (str): Path to save the downsampled segmentation image.

    Returns:
        None
    """
    os.system(f'flirt -in {img_path} -ref {ref_path} -applyisoxfm 1.5 -nosearch -out {out_seg_path} -interp nearestneighbour')

def DownsampleSegs(p, mov, fix, res, fold, version):
    """
    Downsample segmentation files for a given pair of images using FlirtDownsampling.

    Parameters:
        - p (str): Identifier for the patient number.
        - mov (str): Identifier for the moving image.
        - fix (str): Identifier for the fixed image.
        - res (str): Resolution specification.
        - fold (str): Identifier for the fold.
        - version (str): Registration version ('NoReg', 'RigidReg', or 'RigidAffineReg').

    Returns:
        Tuple[str, str]: Paths to the downsampled segmentation files for the moving and fixed images.
    """
    # Rescale to wanted dimensions
    if version == 'NoReg':
        fix_ref_path = f"/media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/{p}/{fix}/wf/brainextraction/{fix}_dtype.nii.gz"
        fix_seg_path = f"/media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/{p}/{fix}/wf/brainextraction/{fix}_seg.nii.gz"
        fix_out_seg = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_resc_new.nii.gz"
        FlirtDownsampling(fix_seg_path, fix_ref_path, fix_out_seg)
        # For moving segmentation
        mov_ref_path = f"/media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/{p}/{mov}/wf/brainextraction/{mov}_dtype.nii.gz"
        mov_seg_path = f"/media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/{p}/{mov}/wf/brainextraction/{mov}_seg.nii.gz"
        mov_out_seg = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{mov}_resc_new.nii.gz"
        FlirtDownsampling(mov_seg_path, mov_ref_path, mov_out_seg)
        return mov_out_seg, fix_out_seg
    elif version == 'RigidReg':
        fix_ref_path = f"/media/andjela/SeagatePor/PairReg/rigid/images/{p}_{mov}_{fix}/{fix}_dtype.nii.gz"
        fix_seg_path = f"/media/andjela/SeagatePor/PairReg/rigid/images/{p}_{mov}_{fix}/{fix}_seg.nii.gz"
        fix_out_seg = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_resc_new.nii.gz"
        FlirtDownsampling(fix_seg_path, fix_ref_path, fix_out_seg)
        # For moving segmentation
        mov_ref_path = f"/media/andjela/SeagatePor/PairReg/rigid/images/{p}_{mov}_{fix}/{mov}_dtype.nii.gz"
        mov_seg_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{mov}.nii.gz"
        mov_out_seg = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{mov}_resc_new.nii.gz"
        FlirtDownsampling(mov_seg_path, mov_ref_path, mov_out_seg)
        return mov_out_seg, fix_out_seg
    else:
        fix_ref_path = f"/media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{mov}_{fix}/{fix}_dtype.nii.gz"
        fix_seg_path = f"/media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{mov}_{fix}/{fix}_seg.nii.gz"
        fix_out_seg = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_resc_new.nii.gz"
        FlirtDownsampling(fix_seg_path, fix_ref_path, fix_out_seg)

        # For moving segmentation
        mov_ref_path = f"/media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{mov}_{fix}/{mov}_dtype.nii.gz"
        mov_seg_path = f"/media/andjela/SeagatePor/PairReg/rigid_affine_corr/images/{p}_{mov}_{fix}/{mov}_seg.nii.gz"
        mov_out_seg = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{mov}_resc_new.nii.gz"
        FlirtDownsampling(mov_seg_path, mov_ref_path, mov_out_seg)
        return mov_out_seg, fix_out_seg
    
def ChangeAffine(pre_input_path, input_path, out_path):
    """
    Changes and saves the affine transformation matrix of a NIfTI image to match another image's affine.

    Parameters:
        - pre_input_path (str): Path to the reference NIfTI image file.
        - input_path (str): Path to the input NIfTI image file with the desired affine.
        - out_path (str): Path to save the NIfTI image with the updated affine.

    Returns:
        None
    """
    img_pre_network = nib.load(pre_input_path)
    data_pre_network = img_pre_network.get_fdata()
    img_input_network = nib.load(input_path)
    data_input_network = img_input_network.get_fdata()

    img_new = nib.Nifti1Image(data_pre_network, img_input_network.affine)
    
    nib.save(img_new, out_path)

def DDFReshaping(dl_ddf_path, trans_ddf_path):
    """
    Reshape and manipulate a 4D medical image in NIfTI format to 5D itk format compatible with ANTs for future analysis.

    Parameters:
        - dl_ddf_path (str): Path to the input NIfTI dense displacement field (DDF) file obtained with DL-based registration.
        - trans_ddf_path (str): Path to save the transformed DDF NIfTI file.

    Returns:
        None
    """

    ddf_img = nib.load(dl_ddf_path)
    data = np.array(ddf_img.get_fdata())
    data = data[:, :, :, np.newaxis,:]
    # Change displacement direction to match ANTs/itk LPS+ system
    data[:, :, :, :, 0] =  -1 * data[:, :, :, :, 0] 
    data[:, :, :, :, 1] =  -1 * data[:, :, :, :, 1]

    hdr = ddf_img.header.copy()
    hdr.set_intent('vector', (), '')

    img = nib.Nifti1Image(data, ddf_img.affine, hdr)
    # Save modified DDF which is now 153,153,125,1,3 and not 153,153,125,3 for future analysis
    nib.save(img, trans_ddf_path)

def DiceScoresDLSynthSeg():
    """
    Function to calculate Dice Scores for a given ANTs initialization version (NoReg, RigidReg or RigidAffineReg) by first warping synthseg 
    segmentations using transformed DDFs obtained by DL-based registration into ANTs composite transform files.
    To be called with warp-and-dice-synthseg.sh file. 
    """
    version = str(sys.argv[1]) #e.g. "RigidReg"
    time_stamp = str(sys.argv[2]) #e.g. "20210527-222928_f0"
    res = str(sys.argv[3]) #e.g. Resolution 1.5, 2.0 (in mm)
    
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

    # Create an empty DataFrame
    columns = ['pair_number', 'scanIDs', 'label', 'Dice Score']
    df = pd.DataFrame(columns=columns, dtype=object)

    dice_means = []
    reg_times = []

    if not os.path.exists(f"/media/andjela/SeagatePor/warped_segs/SynthSeg/{version}_{res}/"):
        os.mkdir(f"/media/andjela/SeagatePor/warped_segs/SynthSeg/{version}_{res}/")

    for pair_number in pair_numbers:
        idx = [int(s) for s in pair_number.split("_") if s.isdigit()]

        p, mov, fix = FindFolderNamePair(folder_names[idx[0]])

        
        # Downsample seg to 153x153x125 from 230x230x188 (which is the output size for synthseg)
        mov_out_seg, fix_out_seg = DownsampleSegs(p, mov, fix, res, fold, version)

        # Change affines to identity to match DDF inputs
        mov_path = f'/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/moving_image.nii.gz'
        mov_out_path = f'/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/mov_seg.nii.gz'
        ChangeAffine(mov_out_seg, mov_path, mov_out_path)

        fix_path = f'/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/fixed_image.nii.gz'
        fix_out_path = f'/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/fix_seg.nii.gz'
        ChangeAffine(fix_out_seg, fix_path, fix_out_path)

        dl_ddf_path = f'/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/ddf.nii.gz'
        trans_ddf_path = f'/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/new_ddf.nii.gz'
        DDFReshaping(dl_ddf_path, trans_ddf_path)

        # Predict warp and time for DeepReg models
        start = time.time()

        warp_path = f"/media/andjela/SeagatePor/warped_segs/SynthSeg/{version}_{res}/warped_{pair_number}.nii.gz"
        
        os.system(f"antsApplyTransforms --default-value 0 --float 0 " \
            f"--input {mov_out_path} " \
            f"--interpolation GenericLabel " \
            f"--input-image-type 0 " \
            f"--output {warp_path} " \
            f"--reference-image {fix_out_path} " \
            f"--verbose 0 " \
            f"--transform {trans_ddf_path}"
            )
        
        reg_time = time.time() - start
        reg_times.append(reg_time)

        fixed_seg = nib.load(fix_out_path).get_fdata()
        warped_seg = nib.load(warp_path).get_fdata()

        # Following SynthSeg documentation where numbers for segmented regions do not follow normal numbering
        labels = np.array([0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60])
        # 0 only represents the backgroung, hence it is excluded
        dice_scores = dice(warped_seg, fixed_seg, labels=labels, include_zero=False)
        
        dice_means.append(np.mean(dice_scores))

        labels = np.delete(labels, np.argwhere(labels == 0))
        for label in labels:
            row = {'pair_number': pair_number, 'scanIDs': f'{mov}_{fix}', 'label': label, 'Dice Score': dice_scores[np.where(labels == label)[0]][0]}
            df = df.append(row, ignore_index=True) 

        print('Pair nb %s   scanIDs %s    Reg Time: %.4f    Dice: %.4f +/- %.4f' % (pair_number, f'{mov}_{fix}', reg_time,
                                                                    np.mean(dice_scores),
                                                                    np.std(dice_scores)))

    if not os.path.exists(f"/media/andjela/SeagatePor/dice_scores/SynthSeg/{version}_{res}_{fold}/"):
        os.mkdir(f"/media/andjela/SeagatePor/dice_scores/SynthSeg/{version}_{res}_{fold}/")

    df.to_csv(f"/media/andjela/SeagatePor/dice_scores/SynthSeg/{version}_{res}_{fold}/dice_scores_{version}")
    print()
    print(f'{version}_{fold}')
    print('Avg Reg Time: %.4f +/- %.4f  (skipping first prediction)' % (np.mean(reg_times),
                                                                        np.std(reg_times)))
    print('Avg Dice: %.4f +/- %.4f' % (np.mean(dice_means), np.std(dice_means)))


if __name__ == '__main__':

    # Select which registration technique is tested. ie. DL-based registration or SyN ANTs

    # For warp-and-dice-synthseg.sh
    DiceScoresDLSynthSeg()

    # For warp-and-dice-ANTs-synthseg.sh
    # DiceScoresANTsSynthSeg()
    