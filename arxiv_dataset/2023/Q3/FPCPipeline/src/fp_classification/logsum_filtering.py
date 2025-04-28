"""

Filter lesions based on log-sum of the uncertainty values as shown
in Nair et al. (2018)

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import numpy as np
import os
from argparse import ArgumentParser
from utils.image_utils import read_itk_image
import pandas as pd
import joblib
from segmentation_metrics.lesion_correspondence import *

if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--unc_method', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--unc_type', type=str, default='predictive')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--threshold', type=float, default=0.9999)

    args = parser.parse_args()

    dataset_df = pd.read_pickle(os.path.join(args.root_dir,
                                             args.dataset,
                                             args.unc_method,
                                             'fp_results',
                                             'intensity',
                                             '{}_dataset.pkl'.format(args.mode)))

    images_dir = os.path.join(args.root_dir,
                              args.dataset,
                              args.unc_method,
                              '{}_images'.format(args.mode))


    result_dir = os.path.join(args.root_dir,
                              args.dataset,
                              args.unc_method,
                              'fp_results',
                              args.unc_type)

    if args.mode == 'train':
        logsum_uncertainties = []
        scaling_statistics = {}

    elif args.mode == 'test':
        scaling_statistics = joblib.load(os.path.join(result_dir,
                                                      'logsum_scaling_stats.pkl'))

        global_min = scaling_statistics['global_min']
        logsum_range = scaling_statistics['range']
        print('Global minimum = {}'.format(global_min))
        print('Range = {}'.format(logsum_range))
        tp_before = 0
        fp_before = 0
        fn_before = 0
        tp_after = 0
        fp_after = 0
        fn_after = 0
        lesions_deleted = 0


    pat_dirs = [f.path for f in os.scandir(images_dir) if f.is_dir()]

    diameters = []
    lesion_labels = []
    scaled_logsum = []

    for pat_dir in pat_dirs:
        pat_id = pat_dir.split(os.sep)[-1]
        pat_df = dataset_df[dataset_df['Patient ID'] == pat_id]

        if args.mode == 'test':
            lesion_diameters = pat_df['Maximum3DDiameter'].to_numpy()
            true_labels = pat_df['label'].to_numpy()

        _, binary_lesion_mask = read_itk_image(os.path.join(pat_dir,
                                                            'binary_post_proc_pred.nii.gz'))

        if args.unc_type == 'predictive':
            _, umap_np = read_itk_image(os.path.join(pat_dir,
                                                     'raw_uncertainty_map.nii.gz'))
        elif args.unc_type == 'aleatoric':
            _, umap_np = read_itk_image(os.path.join(pat_dir,
                                                     'raw_aleatoric_uncertainty_map.nii.gz'))
        elif args.unc_type == 'epistemic':
            _, umap_np = read_itk_image(os.path.join(pat_dir,
                                                     'raw_epistemic_uncertainty_map.nii.gz'))

        _, gt = read_itk_image(os.path.join(pat_dir,
                                            'true_mask.nii.gz'))

        if args.mode == 'test':
            # 1. Graph construction
            corr_graph = create_correspondence_graph(seg=binary_lesion_mask, gt=gt)

            # 2. Using the graph, get object slices and labels for each object slice
            lesion_counts_before = count_detections(corr_graph,
                                                    gt=gt,
                                                    seg=binary_lesion_mask)

            tp_before += lesion_counts_before['true positives']
            fp_before += lesion_counts_before['false positives']
            fn_before += lesion_counts_before['false negatives']

            filtered_lesion_mask = binary_lesion_mask.copy()


        lesion_slices = pat_df['slice'].to_numpy()
        n_slices = len(lesion_slices)

        if args.mode == 'train' and n_slices == 0:
            continue

        print('Computing uncertainty logsum for {} lesions from patient {}'.format(n_slices, pat_id))
        for idx in range(n_slices) :
            l_slice = lesion_slices[idx]
            single_lesion_mask = np.zeros_like(binary_lesion_mask)
            single_lesion_mask[l_slice] += binary_lesion_mask[l_slice]

            # The slice MUST contain a detection!!!
            assert(np.amax(single_lesion_mask) == 1)

            lesion_logsum = np.sum(np.log(umap_np[single_lesion_mask == 1] + 1e-5))

            if args.mode == 'train': # We don't know scaling parameters
                logsum_uncertainties.append(lesion_logsum)
            elif args.mode == 'test':
                lesion_logsum_scaled = (lesion_logsum - global_min)/logsum_range

                if lesion_logsum_scaled >= args.threshold: # delete lesion
                    filtered_lesion_mask[l_slice] = 0
                    lesions_deleted += 1

                diameters.append(lesion_diameters[idx])
                lesion_labels.append(true_labels[idx])
                scaled_logsum.append(lesion_logsum_scaled)

        # Done looking at all lesion slices for the patient.
        if args.mode == 'test':
            # 1. Graph construction
            corr_graph = create_correspondence_graph(seg=filtered_lesion_mask,
                                                     gt=gt)

            # 2. Using the graph, get object slices and labels for each object slice
            lesion_counts_after = count_detections(corr_graph,
                                                   gt=gt,
                                                   seg=filtered_lesion_mask)

            tp_after += lesion_counts_after['true positives']
            fp_after += lesion_counts_after['false positives']
            fn_after += lesion_counts_after['false negatives']

    # Done with all patients
    if args.mode == 'train': # Save the statistics for scaling
        logsum_uncertainties = np.array(logsum_uncertainties)
        global_min = np.amin(logsum_uncertainties)
        global_max = np.amax(logsum_uncertainties)
        logsum_range = global_max - global_min
        scaling_statistics['global_min'] = global_min
        scaling_statistics['range'] = logsum_range
        joblib.dump(scaling_statistics, os.path.join(result_dir,
                                                     'logsum_scaling_stats.pkl'))

    elif args.mode == 'test':
        print('Before filtering :: TP = {} FP = {} FN = {}'.format(tp_before, fp_before, fn_before))
        print('After filtering ::  TP = {} FP = {} FN = {}'.format(tp_after, fp_after, fn_after))
        print('Lesions deleted = {}, Difference in false positives = {} Difference in true positives = {}'.format(lesions_deleted,
                                                                                                                  fp_before-fp_after,
                                                                                                                  tp_before-tp_after))

        # Compute precision and recall
        precision_before = tp_before/(tp_before + fp_before)
        recall_before = tp_before/(tp_before + fn_before)
        f1_before = (2*(precision_before*recall_before))/(precision_before + recall_before)

        precision_after = tp_after/(tp_after + fp_after)
        recall_after = tp_after/(tp_after + fn_after)
        f1_after = (2*(precision_after*recall_after))/(precision_after + recall_after)

        # Compute relative FP and TP reduction
        fp_reduced_relative = (fp_before - fp_after)/fp_before
        tp_reduced_relative = (tp_before - tp_after)/tp_before
        precision_increase = (precision_after-precision_before)/precision_before
        recall_reduced = (recall_before-recall_after)/recall_before
        f1_increase = (f1_after-f1_before)/f1_before

        relative_change_dict = {}
        relative_change_dict['FP reduced'] = [fp_reduced_relative]
        relative_change_dict['TP reduced'] = [tp_reduced_relative]
        relative_change_dict['Precision'] = [precision_increase]
        relative_change_dict['Recall'] = [recall_reduced]
        relative_change_dict['F1'] = [f1_increase]

        relative_change_df = pd.DataFrame.from_dict(data=relative_change_dict)
        relative_change_df.to_pickle(os.path.join(result_dir,
                                                     'relative_reduction_logsum_thresh_{}.pkl'.format(args.threshold)))

        print('Precision :: Before = {}, After = {}'.format(precision_before, precision_after))
        print('Recall :: Before = {}, After = {}'.format(recall_before, recall_after))
        print('F1 :: Before = {}, After = {}'.format(f1_before, f1_after))

        # For later analysis
        joblib.dump(diameters, os.path.join(result_dir,
                                            'diameters.pkl'))

        joblib.dump(lesion_labels, os.path.join(result_dir,
                                                'true_labels.pkl'))

        joblib.dump(scaled_logsum, os.path.join(result_dir,
                                                'scaled_logsum_{}.pkl'.format(args.threshold)))
