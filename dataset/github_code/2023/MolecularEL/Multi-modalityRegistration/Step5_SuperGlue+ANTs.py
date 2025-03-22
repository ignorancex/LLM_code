import glob
import os

import cv2
import nibabel as nib
import numpy as np
import pandas as pd

from superglue_ants_registration_onpair_new_RGB import register_a_pair, register_a_pair_intensitychange, register_3D


# based on https://github.com/ANTsX/ANTs/wiki/Forward-and-inverse-warps-for-warping-images,-pointsets-and-Jacobians
# and
import re

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def get_df(fixed_bbox):
    df = pd.DataFrame(columns=['x', 'y', 't', 'label'])
    row = 0
    for fi in range(len(fixed_bbox)):
        bbox = fixed_bbox[fi]
        assert len(bbox) == 4

        df.loc[row] = [bbox[0], bbox[1], fi, 0]
        row = row + 1
        df.loc[row] = [bbox[2], bbox[1], fi, 0]
        row = row + 1
        df.loc[row] = [bbox[2], bbox[3], fi, 0]
        row = row + 1
        df.loc[row] = [bbox[0], bbox[3], fi, 0]
        row = row + 1

    return df


def mask_to_box(fixed_mask_jpg):
    img = cv2.imread(fixed_mask_jpg)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    resize_box = [cmin, rmin, (cmax - cmin), (rmax - rmin)]

    return resize_box


if __name__ == "__main__":

    # case_folder = '/Data2/Auto_label/IHC+PAS_png'
    case_folder = '/Data/MolecularEL/PAS+Endo_png'
    cases = glob.glob(os.path.join(case_folder, '*'))
    now_middle_idx = 1

    for ci in range(0,len(cases)):
    #for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)

        image_input_dir = os.path.join(case, '5X')
        image_output_root_dir = os.path.join(case, 'ANTs_affine')
        #
        # image_input_dir = '/Data/fromHaichun/tracking_pairwise/slices_all'
        # image_mask_dir = '/Data/fromHaichun/tracking_pairwise/mask_all
        # image_output_root_dir = '/Data/fromHaichun/major_review/registration_all_superglue'

        overall_results_dir = os.path.join(image_output_root_dir, 'all_results')

        slice_files = glob.glob(os.path.join(image_input_dir, '*'))
        slice_files.sort(key=natural_keys)

        image_output_dir = image_output_root_dir

        print('now is %s' % (now_case))
        moving_jpg = glob.glob(os.path.join(image_input_dir, '*IHC.png'))[0]
        fixed_jpg = glob.glob(os.path.join(image_input_dir, '*PAS.png'))[0]

        working_dir = os.path.join(image_output_dir, 'IHC-to-PAS')

        if os.path.exists(os.path.join(working_dir, 'step2_run_ants_reg','output0GenericAffine.mat')):
            continue

        register_a_pair_intensitychange(moving_jpg, fixed_jpg, working_dir, 5)
