"""
Script that extracts lesion patches from uncertainty maps

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import sys
sys.path.append(os.path.join(os.expanduser('~'), 'false_positive_classification_pipeline', 'utils' ))
sys.path.append(os.path.join(os.expanduser('~'), 'false_positive_classification_pipeline', 'seg_metrics' ))
from helper_functions import *
from lesion_correspondence import *
from argparse import ArgumentParser
import random
import pandas as pd
import pickle
from extract_features import *
from image_utils import read_itk_image


def create_patch_dict_for_patient(pat_dir, mode='predictive', data_source='train', dataset='lits'):
    """
    Create dictionary for a patient containing patches from
    the uncertainty map along with label (fp/tp) and related
    patient-level meta-data

    """

    patient_dict = {}
    pat_id = pat_dir.split('/')[-1]

    # Ground Truth
    gt_itk, gt_np = read_itk_image(os.path.join(pat_dir, 'true_mask.nii.gz'))

    # Get ITK metadata (required for radiomics feature extraction)
    patient_dict['Direction'] = gt_itk.GetDirection()
    patient_dict['Spacing'] = gt_itk.GetSpacing()
    patient_dict['Origin'] = gt_itk.GetOrigin()

    print('Meta-data:: Spacing : {}, Direction: {}, Origin: {}'.format(patient_dict['Spacing'], patient_dict['Direction'], patient_dict['Origin']))

    # Predicted binary segmentation
    _ , seg_np = read_itk_image(os.path.join(pat_dir, 'binary_post_proc_pred.nii.gz'))

    # Entropy Map
    if mode == 'predictive' or mode == 'shape':
        try:
            umap_itk, umap_np = read_itk_image(os.path.join(pat_dir, 'raw_uncertainty_map.nii.gz'))
        except:
            umap_itk, umap_np = read_itk_image(os.path.join(pat_dir, 'uncertainty_map.nii.gz'))
    elif mode == 'aleatoric':
            umap_itk, umap_np = read_itk_image(os.path.join(pat_dir, 'raw_aleatoric_uncertainty_map.nii.gz'))
    elif mode == 'epistemic':
        umap_itk, umap_np = read_itk_image(os.path.join(pat_dir, 'raw_epistemic_uncertainty_map.nii.gz'))
    elif mode == 'intensity':
        if dataset == 'lits':
            umap_itk, umap_np = read_itk_image(os.path.join(pat_dir, 'image.nii.gz'))
        elif dataset == 'umc':
            umap_itk, umap_np = read_itk_image(os.path.join(pat_dir, 'dce.nii.gz'))
            umap_np = np.mean(umap_np, axis=-1) # Avg. over contrast axis
        _, liver_mask_np = read_itk_image(os.path.join(pat_dir, 'liver_mask.nii.gz'))

        umap_np = umap_np*liver_mask_np
    else:
        raise ValueError('Invalid mode {} not supported'.format(mode))

    # Create per-patient dict
    print('Creating lesion correspondence graph for Patient {}'.format(pat_id))

    # 1. Graph construction
    corr_graph = create_correspondence_graph(seg=seg_np, gt=gt_np)

    # 2. Using the graph, get object slices and labels for each object slice
    lesion_counts_dict = count_detections(corr_graph,
                                          gt=gt_np,
                                          seg=seg_np)

    # Check if we have 0 detections
    num_pred_lesions = lesion_counts_dict['true positives'] + lesion_counts_dict['false positives']
    print('Number of objects in prediction = {}'.format(num_pred_lesions))
    if num_pred_lesions == 0 or len(lesion_counts_dict['slices']) == 0:
        return None

    # If there are no false positives, return None to counter the imbalance
    if data_source == 'train' and dataset == 'umc':
        if lesion_counts_dict['false positives'] == 0:
            return None

    # 3. Create the dictionary with labels and slices
    assert(umap_np.shape == seg_np.shape)
    patient_dict['pat_id'] = pat_id
    patient_dict['seg'] = seg_np
    patient_dict['gt'] = gt_np
    patient_dict['umap'] = umap_np
    patient_dict['slices'] = lesion_counts_dict['slices']
    patient_dict['labels'] = lesion_counts_dict['labels']

    return patient_dict



def create_features_dataset(pat_dirs, out_dir=None, mode='pred', dataset='lits', data_source='train', paramsFile=None, global_max=1.0, global_min=0.0):
    """
    Creates dataset of patches by creating a
    list of per-patients dictionaries (that contain the data [patch, label] and related meta-data)

    """

    datasets = []

    scratch_dir = os.path.join(out_dir, 'scratch_{}'.format(data_source)) # Temp dir to store intermediate RoIs for feature extraction
    if os.path.exists(scratch_dir) is True:
        shutil.rmtree(scratch_dir)
    os.makedirs(scratch_dir)

    for pat_dir in pat_dirs:

        # Create patient dictionary with patient image(s) + metadata
        pat_dict = create_patch_dict_for_patient(pat_dir,
                                                 mode=mode,
                                                 data_source=data_source,
                                                 dataset=dataset)

        if pat_dict is not None:
            # Create per-patient dataframe with features + label + metadata
            pat_df = get_per_patient_features(pat_dict,
                                              scratch_dir=scratch_dir,
                                              dataset=dataset,
                                              paramsFile=paramsFile,
                                              global_max=global_max,
                                              global_min=global_min)
            # Append datataframe to list
            datasets.append(pat_df)

    # Concatenate all patient dataframes into a single dataset dataframe
    dataset_df = pd.concat(datasets, ignore_index=True)


    return dataset_df

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, default='predictive')
    parser.add_argument('--dataset', type=str, default='lits')
    parser.add_argument('--data_source', type=str, default='train')
    parser.add_argument('--params', type=str, default=None)
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
    print('Start feature computation for {} patients'.format(len(pat_dirs)))

    # Get max, min values
    try:
        patch_statistics = pd.read_pickle(os.path.join(out_dir, 'patch_statistics.pkl'))
        global_max = patch_statistics['max']
        global_min = patch_statistics['min']
    except FileNotFoundError:
        global_max = 1.0
        global_min = 0.0


    if args.params is None:
        paramsFile = os.path.join(out_dir, 'radiomics_params.yaml')
    else:
        paramsFile = args.params

    print('Radiomics parameters : {}'.format(paramsFile))

    # Save the dataset as a CSV
    dataset_df = create_features_dataset(pat_dirs,
                                         mode=args.mode,
                                         out_dir=out_dir,
                                         dataset=args.dataset,
                                         data_source=args.data_source,
                                         global_max=global_max,
                                         global_min=global_min,
                                         paramsFile=paramsFile)

    num_tps = dataset_df[dataset_df['label'] == 0].shape[0]
    num_fps = dataset_df[dataset_df['label'] == 1].shape[0]
    print('Number of TPs = {}, Number of FPs = {}'.format(num_tps, num_fps))

    imbalance = min(num_tps/num_fps, num_fps/num_tps)
    print('Imbalance in dataset = {}'.format(imbalance))

    fname = os.path.join(out_dir, '{}_dataset.pkl'.format(args.data_source))
    dataset_df.to_pickle(fname)


