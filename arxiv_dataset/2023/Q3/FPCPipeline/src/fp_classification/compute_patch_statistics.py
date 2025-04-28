"""

RUN THIS SCRIPT BEFORE PATCH EXTRACTION

Functions:
    1. Compute global min-max over all patches (in the training set)
    2. Compute Range feature to decide bin-width used

"""
import sys
sys.path.append(os.path.join(os.expanduser('~'), 'false_positive_classification_pipeline', 'utils' ))
import os
import numpy as np
import SimpleITK as sitk
from extract_features import *
from argparse import ArgumentParser
import joblib
from patch_extraction import create_features_dataset
from image_utils import read_itk_image

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--mode', type=str, default='predictive')
    parser.add_argument('--dataset', type=str, default='lits')
    args = parser.parse_args()

    # Create the appropriate output directory
    expanded_path = args.images_dir.split('/')

    if expanded_path[-1] == '':
        list_limit = -2
    else:
        list_limit = -1

    base_dir = '/'.join(expanded_path[:list_limit])
    print(base_dir)
    out_dir = os.path.join(base_dir, 'fp_results', args.mode)
    print(out_dir)

    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)

    pat_dirs = [f.path for f in os.scandir(args.images_dir) if f.is_dir()]

    # Part-I :
    # Compute global min-max
    global_max = -1
    global_min = 1e8

    for pat_dir in pat_dirs:
        if args.mode == 'predictive':
            unc_map_filename = os.path.join(pat_dir, 'raw_uncertainty_map.nii.gz')
        elif args.mode == 'intensity':
            if args.dataset == 'lits':
                unc_map_filename = os.path.join(pat_dir, 'image.nii.gz')
            elif args.dataset == 'umc':
                unc_map_filename = os.path.join(pat_dir, 'dce.nii.gz')

            liver_mask_filename = os.path.join(pat_dir, 'liver_mask.nii.gz')

        else:
            unc_map_filename = os.path.join(pat_dir, 'raw_{}_uncertainty_map.nii.gz'.format(args.mode))

        unc_map_itk, unc_map_np = read_itk_image(unc_map_filename)


        if args.mode == 'intensity':
            liver_mask_itk, liver_mask_np = read_itk_image(liver_mask_filename)

            if args.dataset == 'umc':
                unc_map_np = np.mean(unc_map_np, axis=-1)

            # Mask out the liver
            unc_map_np = unc_map_np*liver_mask_np

            # Max and min values inside the liver region
            max_val = np.amax(unc_map_np[liver_mask_np==1])
            min_val = np.amin(unc_map_np[liver_mask_np==1])

        else:
            max_val = np.amax(unc_map_np)
            min_val = np.amin(unc_map_np)

        if max_val > global_max:
            global_max = max_val
        if min_val < global_min:
            global_min = min_val

    print(global_max)
    print(global_min)

    image_scaling_stats = {}
    image_scaling_stats['max'] = global_max
    image_scaling_stats['min'] = global_min

    joblib.dump(image_scaling_stats, os.path.join(out_dir, 'patch_statistics.pkl'))

    # Part-II:
    # Compute 'Range'
    # Save the dataset as a CSV
    dataset_df = create_features_dataset(pat_dirs,
                                         mode=args.mode,
                                         out_dir=out_dir,
                                         dataset=args.dataset,
                                         data_source='train',
                                         paramsFile=args.params,
                                         global_max=global_max,
                                         global_min=global_min)

    fname = os.path.join(out_dir, 'dataset_range.pkl')
    dataset_df.to_pickle(fname)








